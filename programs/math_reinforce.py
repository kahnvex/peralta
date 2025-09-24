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
from transformers.generation.utils import GenerationMixin
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

    def _eval(n):
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
    prompt_len: int           # number of prompt tokens
    gen_len: int              # number of generated tokens
    advantage: torch.Tensor   # scalar float32
    reward: torch.Tensor      # scalar float32


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
        """Load data from CSV file."""
        data = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["python_expression"] and row["natural_language"]:
                    data.append(
                        (
                            row["python_expression"].strip(),
                            row["natural_language"].strip().strip('"'),
                        )
                    )
        return cast(list[tuple[str, str]], data)

    def __iter__(self) -> Iterator[Problem]:
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
                {
                    "role": "assistant",
                    "content": "21 * (4 + 1) - 24",
                },
                {
                    "role": "user",
                    "content": natural_lang,
                },
            ]

            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

            yield Problem(
                prompt=cast(str, prompt),
                target=python_expr,
            )


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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=_LEARNING_RATE)

    def step(self, batch: TensorDict) -> TensorDict:
        toks = batch["toks"].to(dtype=torch.long)              # [B, S]
        prompt_len = batch["prompt_len"].to(dtype=torch.long)  # [B]
        gen_len = batch["gen_len"].to(dtype=torch.long)        # [B]
        advantage = batch["advantage"].to(dtype=torch.float32) # [B]

        output = cast(
            CausalLMOutputWithPast, trl_utils.forward(self.model, toks, 0)
        )
        logits = output.logits
        assert logits is not None

        logits = logits[:, :-1, :]          # [B, S-1, V]
        labels = toks[:, 1:]                # [B, S-1]
        log_probs = trl_utils.selective_log_softmax(logits, labels)  # [B, S-1]

        B, S1 = labels.shape
        device = labels.device
        j = torch.arange(S1, device=device).unsqueeze(0).expand(B, -1)  # [B, S-1]
        p = prompt_len.unsqueeze(1)                                     # [B, 1]
        g = gen_len.unsqueeze(1)                                        # [B, 1]

        # Tokens at label position j belong to generated span if (j+1) in [p, p+g)
        gen_mask = ((j + 1) >= p) & ((j + 1) < (p + g))                 # [B, S-1], bool

        weights = advantage.unsqueeze(1).to(log_probs.dtype) * gen_mask.to(log_probs.dtype)

        denom = gen_len.sum().to(log_probs.dtype).clamp_min(1)  # total generated tokens
        loss = -(log_probs * weights).sum() / denom

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return TensorDict({"loss": loss.detach()}, batch_size=())

    def checkpoint(self, step: int) -> None:
        self.model.save_pretrained(os.path.join(_TONKOTSU_CKPT_DIR, f"step_{step:03d}"))


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
        compiled_model = torch.compile(raw_model)
        self.model = cast(GenerationMixin, compiled_model)
        self.executor = ThreadPoolExecutor(max_workers=8)

    def generate(
        self, prompt_toks: Sequence[int], num_samples: int
    ) -> list[torch.Tensor]:
        future = self.executor.submit(self._generate_sync, prompt_toks, num_samples)
        return future.result()

    def _generate_sync(
        self, prompt_toks: Sequence[int], num_samples: int
    ) -> list[torch.Tensor]:
        # Inputs now created on self.device
        input_toks = torch.tensor(
            [list(prompt_toks)] * num_samples, device=self.device, dtype=torch.long
        )
        attention_mask = torch.ones_like(input_toks, device=self.device)

        with self.lock, torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_toks,
                attention_mask=attention_mask,
                max_new_tokens=_MAX_GENERATION_LENGTH,
                do_sample=True,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                top_k=_TOP_K,
            )
        # Return to CPU for Ray serialization
        sequences = cast(torch.Tensor, outputs)
        generations = [(seq[len(prompt_toks) :]) for seq in sequences]
        return [cast(torch.Tensor, g.to("cpu")) for g in generations]

    def cleanup(self) -> None:
        """This function removes the current model from memory and frees up GPU/CPU shared memory on Jetson.

        This prevents OOM errors when loading new models, before the new checkpoint is loaded, this
        function is called.
        """
        print(f"[generator] deleting current model")
        del self.model
        gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[generator] CUDA cache cleared")

        # Additional cleanup for unified memory systems
        gc.collect()  # Run GC again after CUDA cleanup

    def load_checkpoint(self) -> None:
        with self.lock:
            self.cleanup()
            checkpoints = glob.glob(os.path.join(_SKYLAB_CKPT_DIR, "step_*"))
            checkpoint_path = max(checkpoints)

            raw_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=(torch.float16 if self.device.type == "cuda" else None),
                low_cpu_mem_usage=True,
            ).to(self.device)
            compiled_model = torch.compile(raw_model)
            self.model = cast(GenerationMixin, compiled_model)
            print(f"[generator] loaded model onto {self.device}")


def reward_fn(
    target: str,
    generation: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> float:
    # Keep prompt around for debugging.
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
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.all_rewards: list[float] = []
        self.reward_lock = threading.Lock()
        self.is_complete = False

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
            prompt_tokens = self.tokenizer(
                [problem.prompt],
                return_tensors=transformers.TensorType.PYTORCH,  # pyright: ignore[reportPrivateImportUsage]
            ).input_ids[0]
            generations = ray.get(
                generator.generate.remote(prompt_tokens, self.generations_per_prompt)
            )

            rewards = [
                reward_fn(problem.target, g, self.tokenizer, problem.prompt)
                for g in generations
            ]
            with self.reward_lock:
                for reward in rewards:
                    self.all_rewards.append(reward)

            # If the rollouts have no variance in reward, they are not usable.
            if not np.var(rewards):
                continue

            adv = torch.as_tensor(rewards, dtype=torch.float32)
            adv -= adv.mean()

            p_len = int(len(prompt_tokens))
            for generation, reward, a in zip(generations, rewards, adv):
                g_len = int(len(generation))
                merged = torch.cat(
                    (
                        torch.as_tensor(prompt_tokens, dtype=torch.int32),
                        torch.as_tensor(generation, dtype=torch.int32),
                    )
                )
                self.queue.put(
                    Rollout(
                        toks=merged,
                        prompt_len=p_len,
                        gen_len=g_len,
                        advantage=a,
                        reward=torch.as_tensor(reward, dtype=torch.float32),
                    )
                )
                print(
                    f"[generator][worker_{worker_id}] put rollout in queue, queue size: {self.queue.qsize()}"
                )

    def consume(self) -> Iterator[TensorDict]:
        """Build batch with toks + lengths + advantage."""

        for step in range(self.num_steps):
            rollouts = [self.queue.get() for _ in range(self.batch_size)]
            filtered_rewards = [r.reward for r in rollouts]

            with self.reward_lock:
                all_rewards = list(self.all_rewards)
                self.all_rewards.clear()

            batch_toks = pad_sequence(
                [r.toks for r in rollouts], batch_first=True, padding_value=0
            )
            prompt_len = torch.tensor([r.prompt_len for r in rollouts], dtype=torch.int32)
            gen_len = torch.tensor([r.gen_len for r in rollouts], dtype=torch.int32)
            advantage = torch.stack([r.advantage for r in rollouts]).to(torch.float32)

            batch = TensorDict(
                {
                    "toks": batch_toks,          # [B, S]
                    "prompt_len": prompt_len,    # [B]
                    "gen_len": gen_len,          # [B]
                    "advantage": advantage,      # [B]
                },
                batch_size=(batch_toks.shape[0],),
            )

            batch.set_non_tensor(
                "filtered_reward",
                float(np.mean(filtered_rewards)) if filtered_rewards else 0.0,
            )
            batch.set_non_tensor(
                "all_rewards",
                float(torch.mean(torch.tensor(all_rewards))) if all_rewards else 0.0,
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

        ray.get(trainer.checkpoint.remote(step))
        batcher.load_checkpoint()

        train_metrics["step"] = step
        print(f"[trainer] {pprint.pformat(train_metrics)}")

    tensorboard_writer.close()


def create_tensorboard_writer(root_dir: str) -> SummaryWriter:
    model = _MODEL_NAME.lower().replace("/", "_")
    dataset = _DATASET.split(".")[0]
    log_dir = os.path.join(root_dir, f"reinforce_{model}_{dataset}")
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
