import ast
import csv
import gc
import glob
import operator as op
import os
import platform
import pprint
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from typing import Any, Iterator, Sequence, TypeVar, cast

import numpy as np
import ray
import torch
import transformers
from ray.actor import ActorHandle
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin, GenerateDecoderOnlyOutput
from trl.trainer import utils as trl_utils


_NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

_MODEL_NAME = "Qwen/Qwen3-0.6B"
_DATASET = "math_10k.csv"
_LEARNING_RATE = 2e-6
_MAX_GENERATION_LENGTH = 48
_NUM_WORKERS_PER_GENERATOR = 4
_BATCH_SIZE = 48
_NUM_STEPS = 40
_GENERATIONS_PER_PROMPT = 4
_DEBUG_GENERATIONS = True
_QUEUE_MAX_LEN = _BATCH_SIZE // _GENERATIONS_PER_PROMPT
_TONKOTSU_CKPT_DIR = f"/Volumes/skydisk/projects/peralta/checkpoints/{_NOW}"
_SKYLAB_CKPT_DIR = f"/mnt/smb/skydisk/projects/peralta/checkpoints/{_NOW}"
_TEMPERATURE = 0.7
_TOP_P = 0.95
_TOP_K = 40
_MAX_GRAD_NORM: float | None = 0.5
_GRPO_CLIP_EPS = 0.05
_USE_COMPILE_GENERATOR = bool(int(os.environ.get("USE_COMPILE_GENERATOR", "0")))

T = TypeVar("T")
Metrics = dict[str, Any]


OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def eval_math(expr: str) -> Any:
    node = ast.parse(expr, mode="eval").body

    def _eval(n: ast.AST) -> bool | int | float:
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        raise ValueError("Only numbers and + - * / // % ** (unary +/-) are allowed")

    return _eval(node)


@dataclass
class Problem:
    prompt: str
    target: str


@dataclass
class Rollout:
    toks: torch.Tensor        # prompt + generation (1D, int32)
    prompt_len: int           # num prompt tokens
    gen_len: int              # num generated tokens
    advantage: torch.Tensor   # scalar float32 (group-normalized)
    reward: torch.Tensor      # scalar float32
    old_logprobs: torch.Tensor  # (gen_len,) float32, per generated token


class MathDataset:
    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.rng = random.Random(42)
        self.data = self._load_data()

    def _load_data(self) -> list[tuple[str, str]]:
        data = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("python_expression") and row.get("natural_language"):
                    data.append(
                        (
                            row["python_expression"].strip(),
                            row["natural_language"].strip().strip('"'),
                        )
                    )
        if not data:
            raise RuntimeError("No data loaded from CSV.")
        return data

    def __iter__(self) -> Iterator[Problem]:
        # simple in-memory resampling
        while True:
            python_expr, natural_lang = self.rng.choice(self.data)

            chat = [
                {
                    "role": "system",
                    "content": "Given a description of a math problem, output Python to compute it.",
                },
                {
                    "role": "user",
                    "content": "add 4 and 1, multiply that by 21, then subtract 24.",
                },
                {"role": "assistant", "content": "21 * (4 + 1) - 24"},
                {"role": "user", "content": natural_lang},
            ]

            prompt = cast(
                str,
                self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
            )

            yield Problem(prompt=prompt, target=python_expr)


def node_info() -> str:
    return str(
        {
            "node": platform.node(),
            "machine": platform.machine(),
            "system": platform.system(),
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "torch": torch.__version__,
        }
    )


@ray.remote
class Trainer:
    def __init__(self) -> None:
        print(f"[trainer] initializing trainer: {node_info()}")
        self.model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME)
        # Training-time settings
        self.model.config.use_cache = False
        try:
            self.model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[trainer] gradient_checkpointing_enable failed (ok to ignore): {e}")

        torch.set_float32_matmul_precision("high")

        # Optimizer
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=_LEARNING_RATE, fused=True
            )
        except TypeError as err:
            print(f"[trainer] fused AdamW unavailable, falling back: {err}")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=_LEARNING_RATE
            )

    def step(self, batch: TensorDict) -> TensorDict:
        toks = batch["toks"].to(dtype=torch.long)               # [B, S]
        prompt_len = batch["prompt_len"].to(dtype=torch.long)   # [B]
        gen_len = batch["gen_len"].to(dtype=torch.long)         # [B]
        advantage = batch["advantage"].to(dtype=torch.float32)  # [B]
        old_lp_gen = batch["old_logprobs"].to(dtype=torch.float32)  # [B, Gmax] padded

        # Forward current policy
        output = cast(CausalLMOutputWithPast, trl_utils.forward(self.model, toks, 0))
        logits = output.logits
        assert logits is not None

        logits = logits[:, :-1, :]           # [B, S-1, V]
        labels = toks[:, 1:]                 # [B, S-1]
        lp_cur = trl_utils.selective_log_softmax(logits, labels)  # [B, S-1]

        B, S1 = labels.shape
        device = labels.device

        # Build "generated token" mask at label positions (as in earlier refactor)
        j = torch.arange(S1, device=device).unsqueeze(0).expand(B, -1)  # label index
        p = prompt_len.unsqueeze(1)                                     # [B,1]
        g = gen_len.unsqueeze(1)                                        # [B,1]
        gen_mask = ((j + 1) >= p) & ((j + 1) < (p + g))                 # [B, S-1], bool

        # Build a dense old_logprobs aligned with label positions from the compact per-generated tensor.
        # old_lp_gen is [B, Gmax], where valid columns are 0..g-1 for each row.
        Gmax = old_lp_gen.size(1)
        idx_base = torch.arange(Gmax, device=device).unsqueeze(0).expand(B, -1)  # [B, Gmax]
        pos = (p - 1) + idx_base                                                 # target label positions
        pos_clamped = pos.clamp(min=0, max=max(S1 - 1, 0))
        valid = (idx_base < g) & (pos < S1)                                      # [B, Gmax]
        values = old_lp_gen * valid.to(old_lp_gen.dtype)                          # zero out invalid

        old_lp_dense = torch.zeros((B, S1), device=device, dtype=old_lp_gen.dtype)
        # scatter_add along dim=1; invalid positions add zeros
        old_lp_dense.scatter_add_(1, pos_clamped, values)

        # PPO-style ratio & clipped objective only on generated tokens
        log_ratio = (lp_cur - old_lp_dense).clamp(-10, 10)
        ratio = torch.exp(log_ratio)

        A = advantage.unsqueeze(1)  # [B,1]
        unclipped = ratio * A
        clipped = torch.clamp(ratio, 1.0 - _GRPO_CLIP_EPS, 1.0 + _GRPO_CLIP_EPS) * A
        policy_obj = torch.minimum(unclipped, clipped)

        # === NEW: Reference forward & Schulman unbiased KL (ref over current) ===
        # Forward reference model (no grad) to get per-token log-probs at the chosen labels
        with torch.no_grad():
            ref_out = cast(CausalLMOutputWithPast, trl_utils.forward(self.ref_model, toks, 0))
            ref_logits = ref_out.logits[:, :-1, :]                               # [B, S-1, V]
        lp_ref = trl_utils.selective_log_softmax(ref_logits, labels)             # [B, S-1]
    
        # Compute unbiased per-token estimator: (p_ref/p_cur) - log(p_ref/p_cur) - 1
        # where p_ref/p_cur = exp(lp_ref - lp_cur). (Use a clamp for extreme ratios.)
        log_ratio_ref_cur = (lp_ref - lp_cur).clamp(-20, 20)
        ratio_ref_cur = torch.exp(log_ratio_ref_cur)
    
        kl_unbiased_tok = ratio_ref_cur - log_ratio_ref_cur - 1.0                # [B, S-1]
    
        # Mask to generated positions and average per generated token
        denom = gen_len.sum().to(lp_cur.dtype).clamp_min(1)
        kl_mean = (kl_unbiased_tok * gen_mask_f).sum() / denom
    
        # --- final objective: maximize policy_obj, penalize KL ---
        # If you already have a hyperparam, replace  _KL_BETA with it or pull from self.
        beta = getattr(self, "kl_beta", _KL_BETA)
        loss = -(policy_obj * gen_mask_f).sum() / denom + beta * kl_mean
        
        gen_mask_f = gen_mask.to(lp_cur.dtype)
        denom = gen_len.sum().to(lp_cur.dtype).clamp_min(1)
        loss = -(policy_obj * gen_mask_f).sum() / denom

        # metrics
        approx_kl_unclamped = torch.exp(lp_cur - old_lp_dense) - 1.0 - (lp_cur - old_lp_dense)
        approx_kl = (approx_kl_unclamped * gen_mask_f).sum() / denom
        clip_hi = ratio > (1.0 + _GRPO_CLIP_EPS)
        clip_lo = ratio < (1.0 - _GRPO_CLIP_EPS)
        clipfrac = ((clip_hi | clip_lo).float() * gen_mask_f).sum() / denom

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if _MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), _MAX_GRAD_NORM)

        self.optimizer.step()

        lp_old_mean = (old_lp_dense * gen_mask_f).sum() / denom
        lp_cur_mean = (lp_cur * gen_mask_f).sum() / denom

        return TensorDict(
            {
                "loss": loss.detach(),
                "policy_loss": loss.detach(),
                "approx_kl": approx_kl.detach(),
                "clipfrac": clipfrac.detach(),
                "lp_old_mean": lp_old_mean.detach(),
                "lp_cur_mean": lp_cur_mean.detach(),
            },
            batch_size=(),
        )

    def checkpoint(self, step: int) -> None:
        out_dir = os.path.join(_TONKOTSU_CKPT_DIR, f"step_{step:03d}")
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        print(f"[trainer] checkpoint saved: {out_dir}")


@ray.remote
class Generator:
    def __init__(self) -> None:
        print(f"[generator] initializing generator: {node_info()}")
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        raw_model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype=(torch.float16 if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        ).to(self.device)
        raw_model.eval()
        self.model = torch.compile(raw_model)
        self.executor = ThreadPoolExecutor(max_workers=8)

    def generate(
        self, prompt_toks: Sequence[int], num_samples: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        future = self.executor.submit(self._generate_sync, prompt_toks, num_samples)
        return future.result()

    def _generate_sync(
        self, prompt_toks: Sequence[int], num_samples: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        prompt_toks = list(prompt_toks)
        input_tokens = torch.tensor(
            [prompt_toks] * num_samples, device=self.device, dtype=torch.long
        )
        attention_mask = torch.ones_like(input_tokens, device=self.device)

        with self.lock, torch.no_grad():
            gen = cast(GenerationMixin, self.model).generate(
                input_ids=input_tokens,
                attention_mask=attention_mask,
                max_new_tokens=_MAX_GENERATION_LENGTH,
                do_sample=True,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                top_k=_TOP_K,
                return_dict_in_generate=True,
            )

            prompt_len = input_tokens.size(1)
            gen_out = cast(GenerateDecoderOnlyOutput, gen)
            full = gen_out.sequences  # (B, prompt_len + T)
            new_tokens = full[:, prompt_len:]  # (B, T)
            out2 = cast(CausalLMOutputWithPast, self.model(full))
            assert out2.logits is not None
            logits_full = out2.logits  # (B, prompt_len + T, V)

        next_logits = logits_full[:, :-1, :]
        targets = full[:, 1:]
        logprobs_full = torch.log_softmax(next_logits, dim=-1)
        gathered = logprobs_full.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        old_lp = gathered[:, prompt_len - 1 :]  # only generated part, shape (B, T)

        # Move to CPU for Ray serialization
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(new_tokens.size(0)):
            out.append(
                (
                    new_tokens[i].to("cpu"),
                    cast(torch.Tensor, old_lp[i].to(torch.float32).to("cpu")),
                )
            )
        return out

    def cleanup(self) -> None:
        print(f"[generator] deleting current model")
        try:
            del self.model
        except Exception:
            pass
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[generator] CUDA cache cleared")
        gc.collect()

    def load_checkpoint(self) -> None:
        with self.lock:
            self.cleanup()
            checkpoints = glob.glob(os.path.join(_SKYLAB_CKPT_DIR, "step_*"))
            if not checkpoints:
                print("[generator] no checkpoints found to load")
                return
            checkpoint_path = max(checkpoints)  # latest by lexicographic step
            base = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=(torch.float16 if self.device.type == "cuda" else None),
                low_cpu_mem_usage=True,
            ).to(self.device)
            base.eval()
            self.model = torch.compile(base) if _USE_COMPILE_GENERATOR else base
            print(f"[generator] loaded model onto {self.device} from {checkpoint_path}")


def reward_fn(
    target: str,
    generation: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> float:
    del prompt
    generated_text = tokenizer.decode(
        list(generation), skip_special_tokens=True
    ).strip()
    target_val = None
    try:
        target_val = eval_math(target)
        generated_val = eval_math(generated_text)
    except Exception:
        generated_val = None

    if _DEBUG_GENERATIONS and target_val == generated_val:
        print(f"✅ [reward] {target} == {target_val} == {generated_text}")
    elif _DEBUG_GENERATIONS:
        print(
            f"❌ [reward] {target} == {target_val} != {generated_text} == {generated_val}"
        )

    return 1.0 if target_val == generated_val else 0.0


class Batcher:
    def __init__(
        self,
        generators: Sequence[ActorHandle],
        tokenizer: PreTrainedTokenizerBase,
        generations_per_prompt: int,
        num_workers_per_generator: int,
        dataset: MathDataset,
        batch_size: int,
        num_steps: int,
    ) -> None:
        self.generators = generators
        self.tokenizer = tokenizer
        self.generations_per_prompt = generations_per_prompt
        self.queue = Queue(maxsize=_QUEUE_MAX_LEN)
        self.dataset = dataset
        self.num_workers_per_generator = num_workers_per_generator
        self.threads: list[threading.Thread] = []
        self.is_complete = False
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.all_rewards: list[float] = []
        self.reward_lock = threading.Lock()

    def produce(self) -> None:
        for generator in self.generators:
            for worker_id in range(self.num_workers_per_generator):
                thread = threading.Thread(
                    target=self._produce,
                    args=(generator, worker_id),
                    daemon=True,
                )
                thread.start()
                self.threads.append(thread)

    def load_checkpoint(self) -> None:
        ray.get([g.load_checkpoint.remote() for g in self.generators])

    def _produce(self, generator: ActorHandle, worker_id: int) -> None:
        dataset_iter = iter(self.dataset)
        while not self.is_complete:
            problem = next(dataset_iter)
            prompt_toks = self.tokenizer(
                [problem.prompt],
                return_tensors=transformers.TensorType.PYTORCH,  # pyright: ignore[reportPrivateImportUsage]
            ).input_ids[0]

            results = ray.get(
                generator.generate.remote(prompt_toks, self.generations_per_prompt)
            )
            gens, old_lps = zip(*results)

            rewards = [
                reward_fn(problem.target, g, self.tokenizer, problem.prompt)
                for g in gens
            ]
            with self.reward_lock:
                self.all_rewards.extend(rewards)

            # Ensure GRPO has group variance; otherwise skip
            if not np.var(rewards):
                continue

            # Group-normalized advantages (scalar per sequence)
            adv = torch.as_tensor(rewards, dtype=torch.float32)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            p_len = int(len(prompt_toks))
            for generation, reward, adv_i, old_lp in zip(gens, rewards, adv, old_lps):
                g_len = int(len(generation))
                merged = torch.cat(
                    (
                        torch.as_tensor(prompt_toks, dtype=torch.int32),
                        torch.as_tensor(generation, dtype=torch.int32),
                    )
                )
                self.queue.put(
                    Rollout(
                        toks=merged,
                        prompt_len=p_len,
                        gen_len=g_len,
                        advantage=adv_i,
                        reward=torch.as_tensor(reward, dtype=torch.float32),
                        old_logprobs=old_lp.to(torch.float32),
                    )
                )
                print(
                    f"[generator][worker_{worker_id}] put rollout in queue, queue size: {self.queue.qsize()}"
                )

    def consume(self) -> Iterator[TensorDict]:
        for step in range(self.num_steps):
            rollouts = [self.queue.get() for _ in range(self.batch_size)]
            filtered_rewards = [float(r.reward) for r in rollouts]

            with self.reward_lock:
                all_rewards = list(self.all_rewards)
                self.all_rewards.clear()

            batch_toks = pad_sequence(
                [r.toks for r in rollouts], batch_first=True, padding_value=0
            )  # [B, S]
            prompt_len = torch.tensor([r.prompt_len for r in rollouts], dtype=torch.int32)  # [B]
            gen_len = torch.tensor([r.gen_len for r in rollouts], dtype=torch.int32)        # [B]
            advantage = torch.stack([r.advantage for r in rollouts]).to(torch.float32)      # [B]

            # Pad the per-generated old logprobs to Gmax
            old_lp_padded = pad_sequence(
                [r.old_logprobs for r in rollouts], batch_first=True, padding_value=0.0
            )  # [B, Gmax]

            batch = TensorDict(
                {
                    "toks": batch_toks,
                    "prompt_len": prompt_len,
                    "gen_len": gen_len,
                    "advantage": advantage,
                    "old_logprobs": old_lp_padded,  # per-generated tokens, padded
                },
                batch_size=(batch_toks.shape[0],),  # NOTE: only B, not (B,S)
            )

            batch.set_non_tensor(
                "filtered_reward",
                float(np.mean(filtered_rewards)) if filtered_rewards else 0.0,
            )
            batch.set_non_tensor(
                "all_rewards", float(np.mean(all_rewards)) if all_rewards else 0.0
            )
            batch.set_non_tensor("step", step)
            print(
                f"[trainer] filtered={len(filtered_rewards)}, total={len(all_rewards)}"
            )

            yield batch
        self.is_complete = True


def train(
    trainer: ActorHandle,
    tensorboard_writer: SummaryWriter,
    batcher: Batcher,
) -> None:
    for batch in batcher.consume():
        step = batch["step"]
        tensorboard_writer.add_scalar(
            "filtered_reward", float(batch["filtered_reward"]), step
        )
        tensorboard_writer.add_scalar("all_rewards", float(batch["all_rewards"]), step)

        train_metrics = cast(TensorDict, ray.get(trainer.step.remote(batch)))
        train_metrics = train_metrics.to_dict(convert_tensors=True)

        for metric_name, metric_value in train_metrics.items():
            tensorboard_writer.add_scalar(metric_name, float(metric_value), step)

        # Save checkpoint and refresh generator policies from NAS
        ray.get(trainer.checkpoint.remote(step))
        batcher.load_checkpoint()

        train_metrics["step"] = step
        print(f"[trainer] {pprint.pformat(train_metrics)}")

    tensorboard_writer.close()


def create_tensorboard_writer(root_dir: str) -> SummaryWriter:
    model = _MODEL_NAME.lower().replace("/", "_")
    dataset = _DATASET.split(".")[0]
    log_dir = os.path.join(root_dir, f"grpo_{model}_{dataset}")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def main() -> None:
    runtime_env = {
        "working_dir": ".",
        "py_executable": "uv run --project . --locked",
    }
    ray.init(address="auto", runtime_env=runtime_env)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(project_root, "runs", now)
    os.makedirs(train_dir, exist_ok=True)

    trainer = (
        cast(ActorHandle, Trainer)
        .options(max_restarts=-1, max_task_retries=-1, resources={"head": 1})
        .remote()
    )

    skylab0_generator = (
        cast(ActorHandle, Generator)
        .options(
            max_concurrency=8,
            num_cpus=4,
            resources={"cpu_worker": 1},
            max_restarts=-1,
            max_task_retries=-1,
        )
        .remote()
    )

    skylab1_generator = (
        cast(ActorHandle, Generator)
        .options(
            max_concurrency=8,
            num_gpus=1,
            resources={"gpu_worker": 1},
            max_restarts=-1,
            max_task_retries=-1,
        )
        .remote()
    )

    generators = cast(list[ActorHandle], [skylab0_generator, skylab1_generator])

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    dataset = MathDataset(
        csv_path=os.path.join(project_root, "data", _DATASET),
        tokenizer=tokenizer,
    )
    tensorboard_writer = create_tensorboard_writer(train_dir)

    batcher = Batcher(
        generators=generators,
        tokenizer=tokenizer,
        generations_per_prompt=_GENERATIONS_PER_PROMPT,
        num_workers_per_generator=_NUM_WORKERS_PER_GENERATOR,
        dataset=dataset,
        batch_size=_BATCH_SIZE,
        num_steps=_NUM_STEPS,
    )

    batcher.produce()

    train(
        trainer=cast(ActorHandle, trainer),
        tensorboard_writer=tensorboard_writer,
        batcher=batcher,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
