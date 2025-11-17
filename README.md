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

**LoRA Fine-tuning (fast, parameter-efficient):**

```bash
# Full LoRA training (567 examples, 3 epochs, ~15-20 min)
ALLOW_WANDB=true uv run modal run -m src.core.train \
  --config=config/llama2-lora.yml \
  --data=data/blocksworld_train.jsonl

# Quick LoRA test (7 L1 examples, 1 epoch, ~3-5 min)
ALLOW_WANDB=true uv run modal run -m src.core.train \
  --config=config/llama2-lora-test.yml \
  --data=data/blocksworld_L1_train.jsonl
```

**Full Fine-tuning (all parameters):**

```bash
# Full fine-tuning (567 examples, 3 epochs, ~25-35 min)
GPU_CONFIG=a100-80gb:2 ALLOW_WANDB=true uv run modal run -m src.core.train \
  --config=config/llama2.yml \
  --data=data/blocksworld_train.jsonl

# Quick test (7 L1 examples, 1 epoch, ~5-8 min)
ALLOW_WANDB=true uv run modal run -m src.core.train \
  --config=config/llama2-test.yml \
  --data=data/blocksworld_L1_train.jsonl
```

**Configuration:**

- Base model: `meta-llama/Llama-2-7b-chat-hf`
- Dataset: 567 examples (482 train, 85 val after 15% split)
- LoRA: r=16, α=32, dropout=0.05, LR=2e-4, batch_size=8
- Full (paper): LR=5e-5, global_batch_size=20 (micro=2, grad_accum=10)
- Hardware: 2x A100 GPUs

Models saved to Modal volumes at `/runs/axo-{timestamp}/[lora|full]-out/`.

**Accessing saved models:**

```bash
# List all training runs
uv run modal volume ls example-runs-vol

# View files in a specific run
uv run modal volume ls example-runs-vol axo-2025-11-17-09-14-09-9bf6

# LoRA adapters are in: /runs/{run-name}/lora-out/
# Full fine-tuned models are in: /runs/{run-name}/full-out/
```

### Evaluation

**Baseline (pre-fine-tuning):**

Uses the base `meta-llama/Llama-2-7b-chat-hf` model (before fine-tuning):

```bash
# Evaluate on full L3 test set (1,057 examples)
uv run modal run src.eval_baseline \
  --data-file data/blocksworld_L3_test.jsonl

# Quick test on 50 examples
uv run modal run src.eval_baseline \
  --data-file data/blocksworld_L3_test.jsonl \
  --n-samples 50
```

**Fine-tuned models:**

```bash
# Evaluate LoRA model on full L3 test set (1,057 examples)
uv run modal run src.eval_finetuned \
  --run-name axo-2025-11-17-09-14-09-9bf6 \
  --data-file data/blocksworld_L3_test.jsonl \
  --lora

# Evaluate full fine-tuned model
uv run modal run src.eval_finetuned \
  --run-name axo-2025-11-17-09-20-23-e552 \
  --data-file data/blocksworld_L3_test.jsonl

# Quick test on 50 examples
uv run modal run src.eval_finetuned \
  --run-name <your-run-name> \
  --data-file data/blocksworld_L3_test.jsonl \
  --n-samples 50 \
  --lora  # include this flag for LoRA models
```

**Quick single-example inference test:**

```bash
# Test full fine-tuned model
uv run modal run src.quick_inference_test --run-name axo-2025-11-17-09-20-23-e552

# Test LoRA model
uv run modal run src.quick_inference_test --run-name axo-2025-11-17-09-14-09-9bf6 --lora
```

Results are saved to `results/` directory as both `.json` (machine-readable) and `.txt` (human-readable) files.

## Fine-tuning Results

Evaluation on L3 test set (1,057 examples, 6-step plans):

| Model                       | Exact Match | Notes                         |
| --------------------------- | ----------- | ----------------------------- |
| **Paper (Llama-2-7b-chat)** | 61.0%       | Full fine-tuning, batch=20    |
| **LoRA (ours)**             | **59.7%**   | r=16, α=32, LR=2e-4, batch=8  |
| **Full fine-tuning (ours)** | _pending_   | LR=5e-5, batch=20 (corrected) |

**Key findings:**

- LoRA achieved **59.7% accuracy**, only **1.3% below the paper's result**
- Both models maintain 100% valid plan format (all outputs have proper "Plan:" and "step X:" structure)
- LoRA training: 93 steps, final train_loss=0.015, ~12 minutes on 2x A100-40GB
- Full training: 72 steps, final train_loss=0.020, ~20 minutes on 2x A100-80GB

**Training runs:**

- LoRA: `axo-2025-11-17-09-14-09-9bf6` (https://wandb.ai/aicellio/blocksworld-planning/runs/m8d9n9ia?nw=nwuserholma91)
- Full (corrected): `axo-2025-11-17-12-42-41-3911` (https://wandb.ai/aicellio/blocksworld-planning/runs/cvc2abrx?nw=nwuserholma91)
