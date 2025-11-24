"""
Compute extraction rates for MLP and MHSA components across layers.

This implements Section 5.1 from the paper: "Attention Extract the Answers"

The extraction rate measures whether a component's hidden state at the last token
can directly decode the correct answer without going through the rest of the model.

Usage:
    # Run on full fine-tuned model (uses 400 correct examples)
    uv run modal run src.interpretability.extraction_rates \
        --run-name axo-2025-11-17-12-42-41-3911 \
        --data-file data/blocksworld_L3_correct_full.jsonl

    # Run on LoRA model (uses 400 correct examples)
    uv run modal run src.interpretability.extraction_rates \
        --run-name axo-2025-11-17-09-14-09-9bf6 \
        --data-file data/blocksworld_L3_correct_lora.jsonl \
        --lora
"""

import modal
import json
import re

app = modal.App("extraction-rates-analysis")

model_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("example-runs-vol", create_if_missing=True)

hf_image = modal.Image.debian_slim().pip_install(
    "transformers==4.44.0",
    "torch==2.3.0",
    "accelerate==0.33.0",
    "peft==0.12.0",
)


def extract_answer_tokens_from_plan(plan_text: str) -> list[str]:
    """
    Extract the answer token (last word) from each step in the plan.

    Following the paper's fill-in-the-blank format, we extract the last token
    on each line, which is what the model should predict.

    Args:
        plan_text: Plan text like "Plan:\nstep 1: pick-up red\nstep 2: stack red on-top-of blue"

    Returns:
        List of answer tokens in order, e.g. ["red", "blue"]

    Examples:
        "step 1: pick-up white" → "white"
        "step 2: stack white on-top-of table" → "table"
    """
    answers = []
    lines = plan_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line.startswith("step"):
            continue

        # Extract the last word on the line (the answer token)
        tokens = line.split()
        if len(tokens) > 0:
            answers.append(tokens[-1])

    return answers


def find_answer_token_positions(
    input_ids: list[int],
    expected_answer_ids: list[int],
) -> list[tuple[int, int]]:
    """
    Find the positions of answer tokens in the input sequence.

    Args:
        input_ids: Token IDs from tokenizer
        expected_answer_ids: List of answer token IDs to find (one per step)

    Returns:
        List of (answer_pos, last_token_pos) tuples for each step.
        answer_pos is the position of the answer token.
        last_token_pos is the position right before the answer token.
    """
    positions = []
    answer_occurrence_count = {}

    for step_idx, expected_answer_id in enumerate(expected_answer_ids):
        # Track which occurrence of this answer we're looking for
        if expected_answer_id not in answer_occurrence_count:
            answer_occurrence_count[expected_answer_id] = 0

        target_occurrence = answer_occurrence_count[expected_answer_id]
        answer_occurrence_count[expected_answer_id] += 1

        # Find all positions where this answer token appears
        answer_positions = [
            i for i, tid in enumerate(input_ids) if tid == expected_answer_id
        ]

        assert target_occurrence < len(
            answer_positions
        ), f"Answer token {expected_answer_id} occurrence {target_occurrence} not found"

        answer_pos = answer_positions[target_occurrence]
        last_token_pos = answer_pos - 1

        assert last_token_pos >= 0, f"Answer token at position 0, no previous token"
        positions.append((answer_pos, last_token_pos))

    return positions


def check_extraction_event(
    hidden_state: "torch.Tensor",
    lm_head: "torch.Tensor",
    expected_token_id: int,
) -> bool:
    """
    Check if a hidden state can directly decode to the expected token.

    Args:
        hidden_state: Hidden state vector [hidden_dim]
        lm_head: Unembedding matrix [vocab_size, hidden_dim]
        expected_token_id: Expected token ID

    Returns:
        True if argmax(lm_head @ hidden_state) == expected_token_id
    """
    import torch

    assert (
        len(hidden_state.shape) == 1
    ), f"Expected 1D hidden state, got shape {hidden_state.shape}"

    # Project to vocabulary
    logits = torch.matmul(hidden_state, lm_head.T)  # [vocab_size]
    predicted_token_id = torch.argmax(logits).item()

    return predicted_token_id == expected_token_id


@app.function(
    image=hf_image,
    gpu="a100-80gb",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": model_cache,
        "/runs": runs_volume,
    },
    timeout=10800,  # 3 hours
)
def compute_extraction_rates(run_name: str, examples: list[dict], use_lora: bool):
    """
    Compute extraction rates for MLP, MHSA, and layer outputs across all layers.

    Returns a dictionary with:
        - results: dict with mlp/mhsa/layer extraction rates per layer
        - num_layers: number of layers
        - num_examples: number of examples processed
        - stats: processing statistics
    """
    import os
    import glob
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    model_cache.reload()
    runs_volume.reload()

    os.environ["HF_HOME"] = "/cache"

    if use_lora:
        base_model_id = "meta-llama/Llama-2-7b-chat-hf"
        lora_path = f"/runs/{run_name}/lora-out"

        print(f"Loading base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
    else:
        model_path = f"/runs/{run_name}/full-out"

        checkpoints = sorted(glob.glob(f"{model_path}/checkpoint-*"))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            print(f"Loading from checkpoint: {checkpoint_path}")
            model_path = checkpoint_path

        print(f"Loading full fine-tuned model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(f"/runs/{run_name}/full-out")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model.eval()
    model_cache.commit()

    print(f"Model loaded! Processing {len(examples)} examples...")
    print(f"Model has {model.config.num_hidden_layers} layers")

    # embedding matrix (V x D) (for decoding hidden states)
    lm_head = model.lm_head.weight
    assert lm_head.shape == (32000, 4096), f"Unexpected LM head shape: {lm_head.shape}"

    num_layers = model.config.num_hidden_layers
    extraction_events = {
        "mlp": [[] for _ in range(num_layers)],
        "mhsa": [[] for _ in range(num_layers)],
        "layer": [[] for _ in range(num_layers)],
    }

    stats = {
        "examples_processed": 0,
        "steps_analyzed": 0,
        "steps_with_correct_final_prediction": 0,
    }

    print("Starting extraction rate computation...")

    for ex_idx, ex in enumerate(examples):
        # Construct the full prompt with the expected plan (teacher-forcing)
        # This allows us to analyze how each layer processes the correct answer
        # Note: We're analyzing CORRECT predictions only, pre-filtered via filter_correct_predictions.py
        prompt = f"""[INST] {ex['instruction']}

{ex['input']} [/INST]
{ex['output']}"""

        # tokenize the full prompt, we get back 1xseq_len tensor
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        assert inputs["input_ids"].shape[0] == 1, "Unexpected tokenized shape"

        # Forward pass with teacher-forcing: we pass the full sequence (including correct plan),
        # but the model's causal attention mask ensures that at position i, it can only see
        # tokens 0...i, not future tokens. Example:
        #   Sequence: ["[INST]", ..., "pick-", "up", "blue", "step", ...]
        #   At position "up": hidden state has seen ["[INST]", ..., "pick-", "up"]
        #                     logits predict next token (should be "blue")
        #                     but "blue" is masked out during attention computation
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states  # (embeddings, layer1, ..., layerN)
        assert len(hidden_states) == num_layers + 1, "Unexpected hidden states shape"
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        input_ids = inputs["input_ids"][0].tolist()  # [seq_len]
        assert logits.shape == (1, len(input_ids), 32000), "Unexpected logits shape"

        plan_text = ex["output"]
        expected_answers = extract_answer_tokens_from_plan(plan_text)
        expected_answer_ids = [
            tokenizer.encode(answer, add_special_tokens=False)[0]
            for answer in expected_answers
        ]

        # find positions of these answer tokens in the tokenized sequence
        positions = find_answer_token_positions(input_ids, expected_answer_ids)
        stats["examples_processed"] += 1

        # for each step, check if the model's final prediction is correct
        # we can enumerate over the positions because each step has exactly one answer token at one position
        for step_idx, (answer_pos, last_token_pos) in enumerate(positions):
            stats["steps_analyzed"] += 1

            expected_answer_id = expected_answer_ids[step_idx]

            # get the final model prediction at this position
            final_logits = logits[0, last_token_pos, :]
            predicted_token_id = torch.argmax(final_logits).item()

            # was the final prediction correct? only proceed if it was
            if predicted_token_id != expected_answer_id:
                continue

            stats["steps_with_correct_final_prediction"] += 1

            # compute extraction rates for each layer
            for layer_idx in range(num_layers):
                # hidden_states[0] is embeddings, hidden_states[1] is layer 0, etc.
                h_layer = hidden_states[layer_idx + 1][0, last_token_pos, :]

                # Check if this layer can extract the answer
                # TODO: Separate MLP and MHSA components
                is_extraction = check_extraction_event(
                    h_layer, lm_head, expected_answer_id
                )
                extraction_events["layer"][layer_idx].append(is_extraction)

        if (ex_idx + 1) % 10 == 0:
            print(f"Processed {ex_idx + 1}/{len(examples)} examples...")

    print(f"\n{'='*60}")
    print("Processing Statistics:")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print(f"{'='*60}\n")

    # Compute mean extraction rates per layer
    results = {
        "mlp": [],
        "mhsa": [],
        "layer": [],
    }

    for component in ["mlp", "mhsa", "layer"]:
        for layer_idx in range(num_layers):
            events = extraction_events[component][layer_idx]
            if len(events) > 0:
                rate = sum(events) / len(events)
            else:
                rate = 0.0
            results[component].append(rate)

    return {
        "results": results,
        "num_layers": num_layers,
        "num_examples": len(examples),
        "stats": stats,
    }


@app.local_entrypoint()
def main(
    run_name: str = "axo-2025-11-17-12-42-41-3911",
    data_file: str = "data/blocksworld_L3_correct_full.jsonl",
    n_samples: int = 0,
    lora: bool = False,
):
    """
    Main entry point for extraction rate computation.
    """
    print(f"Loading test data from {data_file}...")
    with open(data_file, "r") as f:
        examples = [json.loads(line) for line in f]

    if n_samples > 0:
        examples = examples[:n_samples]
        print(f"Using first {n_samples} examples (out of {len(examples)} total).\n")
    else:
        print(f"Using all {len(examples)} correct examples for analysis.\n")

    model_type = "LoRA" if lora else "Full fine-tuned"
    print(f"Computing extraction rates on {model_type} model (run: {run_name})...")

    data = compute_extraction_rates.remote(run_name, examples, lora)

    print(f"\n{'='*60}")
    print(f"EXTRACTION RATE RESULTS ({model_type} Model)")
    print(f"{'='*60}")
    print(f"Run name: {run_name}")
    print(f"Dataset: {data_file}")
    print(f"Examples: {data['num_examples']}")
    print(f"Layers: {data['num_layers']}")
    print(f"{'='*60}\n")

    # Print some sample rates
    print("Sample extraction rates (every 4th layer):")
    print(f"{'Layer':<8} {'Layer Output':<15}")
    print("-" * 25)
    for i in range(0, data["num_layers"], 4):
        layer_rate = data["results"]["layer"][i]
        print(f"{i:<8} {layer_rate:.3f}")

    # Save results
    model_name = run_name.replace("axo-", "").replace("/", "_")
    model_suffix = "lora" if lora else "full"
    output_file = f"results/extraction_rates_{model_suffix}_{model_name}.json"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nNext steps:")
    print("1. Implement MLP/MHSA separation using forward hooks")
    print("2. Generate plots (Figure 3/4 style)")
    print("3. Run on both LoRA and full models for comparison")
