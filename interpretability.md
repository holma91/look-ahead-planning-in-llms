# Interpretability

Two different sections; Information flow analysis and Probing internal representations. Look at interpretability.md for details.

Following the paper, we analyze only correctly predicted examples (400 samples from L3 test set).

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

**Goal:** Measure how MLP vs MHSA components extract information at the last token position. We want to show that MHSA output in middle layers can directly decode correct answer tokens (e.g., "red", "blue", "table").

**Definition (from paper):**
An "extraction event" occurs at layer `l` for position `N` (last token) if:

```
argmax(E · h_N^l) = e*
```

where:

- `E` is the unembedding matrix (lm_head)
- `h_N^l` is the hidden state at layer `l`, position `N`
- `e*` is the expected next token (answer token)

The extraction rate for a component is the fraction of steps where an extraction event occurs.

**Implementation details:**

We use **teacher-forcing** to run the model on correct sequences:

- Input: full prompt including the correct plan (e.g., `[INST] ... [/INST]\nPlan:\nstep 1: pick-up red\nstep 2: stack red on-top-of table`)
- The model's causal attention mask ensures position `i` only sees tokens `0...i` (no "looking ahead")
- We analyze hidden states at positions right before each answer token

For each layer, we extract three types of hidden states:

1. **MHSA output**: Raw output from `self_attn` module (before adding to residual)
2. **MLP output**: Raw output from `mlp` module (before adding to residual)
3. **Layer output**: Full layer output after both MHSA and MLP blocks

We capture MHSA/MLP outputs using PyTorch forward hooks registered on the respective modules.

**LLaMA-2 architecture (pre-norm):**

From `model_details/llama-2-7b-chat-hf/model.safetensors.index.json`, each layer has:

```txt
model.layers.0.input_layernorm            ← norm before attention
model.layers.0.self_attn.{q,k,v,o}_proj  ← attention
model.layers.0.post_attention_layernorm   ← norm before MLP
model.layers.0.mlp.{gate,up,down}_proj    ← MLP
```

Layer computation:

```python
# Pre-norm architecture
x = x + self_attn(input_layernorm(x))           # MHSA block
x = x + mlp(post_attention_layernorm(x))        # MLP block
```

We capture:

- `h_mhsa`: output of `self_attn(...)` before residual addition
- `h_mlp`: output of `mlp(...)` before residual addition
- `h_layer`: final output `x` after both blocks

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

**Results (Full fine-tuned model):**

| Component | Peak Rate | Peak Layer | Starts Extracting |
| --------- | --------- | ---------- | ----------------- |
| MHSA      | 70.71%    | Layer 24   | Layer 16          |
| MLP       | 71.97%    | Layer 22   | Layer 17          |
| Layer     | 100%      | Layer 31   | Layer 20 (>90%)   |

**Key observations:**

- ✓ Layer output reaches >90% extraction by layer 20, confirming information is available in middle layers
- ✓ MHSA starts extracting earlier and peaks in middle-upper layers (16-24)
- ✓ Both MHSA and MLP contribute significantly to extraction (70-72% peak rates)
- ⚠️ Our MLP peak (~72%) is higher than paper's (~35-40%), possibly due to different fine-tuning dynamics or dataset variations
- ✓ Qualitative pattern matches: MHSA dominates in middle layers, both components drop off at later layers

The high extraction rates in middle layers (20-24) support the paper's hypothesis that the model has already "decided" the answer tokens well before the final layer.

**Analysis statistics:**

- Examples processed: 400 (L3 correct predictions)
- Total steps analyzed: 2,400 (6 steps per L3 problem)
- Steps with correct final prediction: 1,345 (56% - only these were used for extraction rate calculation)

**Visualizations:**

- Plot: `results/extraction_rates_full_plot.png`
- Notebook: `scripts/analyze_extraction_rates.ipynb`

**Critical analysis:**

The paper claims "MHSA is primarily responsible for answer extraction" based on MHSA >> MLP in their model (~70% vs ~35%). However, this is an **empirical observation of one model instance**, not a theoretical necessity:

1. **No fundamental reason**: Both MHSA and MLP are universal approximators. The extraction task (does vector h point toward the right vocabulary token?) doesn't inherently favor either component.

2. **Training-dependent**: Our model shows MLP ≈ MHSA (~71-72% for both), proving this is not a universal property but rather depends on training dynamics (initialization, learning rates, random seed).

3. **Multiple valid strategies**: The fact that different models learn different computational strategies (MHSA-heavy vs balanced MLP+MHSA) is actually more interesting - it reveals the flexibility of transformer computation rather than a single "correct" mechanism.

**Takeaway**: The paper's mechanistic analysis describes _how their specific model works_, not _how transformers must work_ for planning tasks. Our divergent results strengthen this interpretation.

### Attention source analysis (Section 5.2)

This traces where MHSA gets its information from. We want to show that planning mainly depends on "Goal state spans" and "Recent history steps".

## Probing internal representations

### Future decisions probing

Probe whether middle/upper layers encode short-term future decisions. This is the "look-ahead planning" hypothesis.

### History step causality

Mask/prevent information flow from different history steps and measure impact on final decisions.
