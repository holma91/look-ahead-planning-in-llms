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

# Overview

The project can naturally be categorized into two parts, fine-tuning and interpretability. For details of the fine-tuning process, read fine-tuning.md

## Fine-tuning

Read [fine-tuning.md](fine-tuning.md) for the complete fine-tuning process. As output, we have two trained models:

### Available Models

Both models are stored on Modal volumes at `/runs/{run-name}/[lora|full]-out/`:

1. **LoRA Model**: `axo-2025-11-17-09-14-09-9bf6`

   - Location: `/runs/axo-2025-11-17-09-14-09-9bf6/lora-out/`
   - Performance: 59.7% exact match on L3 test set
   - Training: [WandB run](https://wandb.ai/aicellio/blocksworld-planning/runs/m8d9n9ia?nw=nwuserholma91)

2. **Full Fine-tuned Model**: `axo-2025-11-17-12-42-41-3911`
   - Location: `/runs/axo-2025-11-17-12-42-41-3911/full-out/`
   - Performance: 58.8% exact match on L3 test set
   - Training: [WandB run](https://wandb.ai/aicellio/blocksworld-planning/runs/cvc2abrx?nw=nwuserholma91)

**Accessing models:**

```bash
# List all training runs
uv run modal volume ls example-runs-vol

# View files in the LoRA model
uv run modal volume ls example-runs-vol axo-2025-11-17-09-14-09-9bf6/lora-out

# View files in the full fine-tuned model
uv run modal volume ls example-runs-vol axo-2025-11-17-12-42-41-3911/full-out
```

**Quick inference test:**

```bash
# Test LoRA model
uv run modal run -m src.quick_inference_test --run-name axo-2025-11-17-09-14-09-9bf6 --lora

# Test full fine-tuned model
uv run modal run -m src.quick_inference_test --run-name axo-2025-11-17-12-42-41-3911
```

# Interpretability

Two different sections; Information flow analysis and Probing internal representations.

Following the paper, we analyze **only correctly predicted examples** (400 samples from L3 test set).

**Prepare datasets of correct predictions:**

```bash
# Filter full fine-tuned model results
uv run python scripts/filter_correct_predictions.py \
    --eval-results results/full_2025-11-17-12-42-41-3911.json \
    --output data/blocksworld_L3_correct_full.jsonl \
    --max-samples 400

# Filter LoRA model results
uv run python scripts/filter_correct_predictions.py \
    --eval-results results/lora_2025-11-17-09-14-09-9bf6.json \
    --output data/blocksworld_L3_correct_lora.jsonl \
    --max-samples 400
```

## Information flow analysis

### Extraction rates (Section 5.1)

This measures how MLP vs MHSA components extract information at the last token. We calculate extraction rates to show that MHSA output in the middle layers can decode correct colors/decisions.

**Run extraction rate analysis:**

```bash
# Analyze full fine-tuned model
uv run modal run src.interpretability.extraction_rates \
    --run-name axo-2025-11-17-12-42-41-3911 \
    --data-file data/blocksworld_L3_correct_full.jsonl

# Analyze LoRA model
uv run modal run src.interpretability.extraction_rates \
    --run-name axo-2025-11-17-09-14-09-9bf6 \
    --data-file data/blocksworld_L3_correct_lora.jsonl \
    --lora
```

### Attention source analysis (Section 5.2)

This traces where MHSA gets its information from. We want to show that planning mainly depends on "Goal state spans" and "Recent history steps".

## Probing internal representations

### Future decisions probing

Probe whether middle/upper layers encode short-term future decisions. This is the "look-ahead planning" hypothesis.

### History step causality

Mask/prevent information flow from different history steps and measure impact on final decisions.
