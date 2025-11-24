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

Two different sections; Information flow analysis and Probing internal representations. See [interpretability.md](interpretability.md) for details.

## Analysis Tasks

### Information Flow Analysis

- [x] **Extraction rates (Section 5.1)** - Measure how MLP vs MHSA components extract information at the last token. Results show both components contribute equally (~71-72%) in our model, unlike the paper's MHSA-dominated pattern.
- [ ] **Attention source analysis (Section 5.2)** - Trace where MHSA gets information from (goal states vs history steps).

### Probing Internal Representations

- [ ] **Future decisions probing (Section 6.1)** - Probe whether middle/upper layers encode short-term future decisions using linear and nonlinear probes.
- [ ] **History step causality (Section 6.2)** - Mask/prevent information flow from different history steps and measure impact on final decisions.
