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
# Full training (567 examples, 3 epochs, ~15-20 min)
ALLOW_WANDB=true uv run modal run -m src.core.train \
  --config=config/llama2-blocksworld.yml \
  --data=data/blocksworld_train.jsonl

# Quick test (7 L1 examples, 1 epoch, ~3-5 min)
ALLOW_WANDB=true uv run modal run -m src.core.train \
  --config=config/llama2-blocksworld-test.yml \
  --data=data/blocksworld_L1_train.jsonl
```

**Configuration:**

- Base model: `meta-llama/Llama-2-7b-chat-hf`
- Fine-tuning: LoRA (r=16, α=32, dropout=0.05)
- Training: 3 epochs, LR=2e-4, batch_size=8, 2x A100 GPUs
- Dataset: 567 examples (482 train, 85 val after 15% split)

Models are saved to Modal volumes at `/runs/axo-{timestamp}/lora-out/`.

### Evaluation

**Baseline (pre-fine-tuning):**

The following uses https://huggingface.co/meta-llama/Llama-2-7b-chat-hf:

```bash
uv run modal run src.eval_baseline \
  --data-file data/blocksworld_L3_test.jsonl \
  --n-samples 50
```

**Fine-tuned model:**

```bash
uv run modal run src.eval_finetuned \
  --run-name <your-run-name> \
  --data-file data/blocksworld_L3_test.jsonl \
  --n-samples 50
```

**Quick single-example test:**

```bash
uv run modal run src.quick_test --run-name <your-run-name>
```

Results are saved to `results/` directory.
