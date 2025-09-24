.PHONY: black check-black

sort-imports:
	uv run ruff check . --select I --fix

format:
	uv run ruff format

tensorboard:
	uv run tensorboard --logdir "runs/*/tensorboard/" --port 6006

start-cluster:
	scripts/start_cluster.sh

stop-cluster:
	scripts/stop_cluster.sh

train-reinforce:
	uv run python3 -m programs.math_reinforce

train-grpo:
	uv run python3 -m programs.math_grpo

clean:
	rm -rf runs/*/snapshots/*

clean-all:
	rm -rf runs/*

serve-blog:
	uv run scripts/serve_blog.py

build-blog:
	uv run scripts/serve_blog.py -w
