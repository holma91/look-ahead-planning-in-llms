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

This measures how MLP vs MHSA components extract information at the last token. We calculate extraction rates to show that MHSA output in the middle layers can decode correct colors/decisions.

We can tell from "model_details/llama-2-7b-chat-hf/model.safetensors.index.json" that each layer has:

```txt
model.layers.0.input_layernorm            ← norm before attention
model.layers.0.self_attn.{q,k,v,o}_proj  ← attention
model.layers.0.post_attention_layernorm   ← norm before MLP
model.layers.0.mlp.{gate,up,down}_proj    ← MLP
```

So a LLaMa layer does:

```txt
# Pre-norm architecture
x = x + self_attn(input_layernorm(x))           # MHSA block
x = x + mlp(post_attention_layernorm(x))        # MLP block
```

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
