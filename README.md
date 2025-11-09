# Reproduction of "Unlocking the Future: Exploring Look-Ahead Planning Mechanistic Interpretability in Large Language Models"

Concretely, we:

- **Recreate the synthetic Blocksworld environment** by generating datasets of different difficulty levels (2, 4, and 6-step optimal plans).
- **Fine-tune a 7B-class language model** on the Blocksworld planning task. Specifically, we use Llama-2 7b.
- **Reproduce the core interpretability analyses** from the paper:
  - extraction-rate curves across layers (MHSA vs. MLP),
  - information-flow analysis between input chunks (init, goal, history, action),
  - linear and nonlinear probes for current block states and future decisions.
- **Extend the original work** by:
  - running the same analyses on a modern open-weight 7B model (e.g. Qwen 3.1–7B),
  - increasing the planning horizon (8–10-step plans) to study how look-ahead depth scales with task complexity.

### Installation

We use UV for dependency management https://docs.astral.sh/uv/guides/projects/#managing-dependencies.

### Data generation

To generate the synthetic data used for finetuning: `uv run scripts/generate_blocksworld_data.py`.

The data can be found at at `./data` in three different difficulty levels:

- L1: each plan has 2 steps
- L2: each plan has 4 steps
- L3: each plan has 6 steps

Here is a L2 sample:

```txt
Rule:
You can pick-up color1. You can stack color1 on-top-of color2. You can stack color1 on-top-of table.

Init state:
<green>
<yellow>
<blue on red>

Goal state:
<red on green>
<blue on yellow>

Plan:
step 1: pick-up blue
step 2: stack blue on-top-of yellow
step 3: pick-up red
step 4: stack red on-top-of green
```

### Fine-tuning

We use [Modal](https://modal.com/) + [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for serverless GPU training.

**Setup:**

1. Install: `uv add modal fastapi`
2. Auth: `uv run modal setup`
3. Configure secrets in Modal dashboard:
   - `huggingface-secret` (HF token with Llama-2 access)
   - `wandb-secret` (optional, for training logs)

**Train:**

```bash
ALLOW_WANDB=true uv run modal run -m src.train \
  --config=config/llama2-blocksworld.yml \
  --data=data/blocksworld_train.jsonl
```

Models are saved to Modal volumes at `/runs/axo-{timestamp}/`.
