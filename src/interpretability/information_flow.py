"""
Compute information flow from input chunks to the last token using attention gradients.

This implements Section 5.2 from the paper: "Attention Extract from Goal and History"

The information flow measures which input chunks (init state, goal state, history steps)
MHSA primarily relies on when making predictions.

Usage:
    # Run on full fine-tuned model
    uv run modal run src.interpretability.information_flow \
        --run-name axo-2025-11-17-12-42-41-3911 \
        --data-file data/blocksworld_L3_correct_full.jsonl
        
    # Run on LoRA model
    uv run modal run src.interpretability.information_flow \
        --run-name axo-2025-11-17-09-14-09-9bf6 \
        --data-file data/blocksworld_L3_correct_lora.jsonl \
        --lora
"""

import modal
import json
import re

app = modal.App("information-flow-analysis")

model_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("example-runs-vol", create_if_missing=True)

hf_image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Required for pip install from git repos
    .pip_install(
        "transformers==4.44.0",
        "torch==2.3.0",
        "accelerate==0.33.0",
        "peft==0.12.0",
        "git+https://github.com/davidbau/baukit",
    )
)


def extract_answer_tokens_from_plan(plan_text: str) -> list[str]:
    """
    Extract the answer token (last word) from each step in the plan.

    Args:
        plan_text: Plan text like "Plan:\nstep 1: pick-up red\nstep 2: stack red on-top-of blue"

    Returns:
        List of answer tokens in order, e.g. ["red", "blue"]
    """
    answers = []
    lines = plan_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line.startswith("step"):
            continue

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
    """
    positions = []
    answer_occurrence_count = {}

    for step_idx, expected_answer_id in enumerate(expected_answer_ids):
        if expected_answer_id not in answer_occurrence_count:
            answer_occurrence_count[expected_answer_id] = 0

        target_occurrence = answer_occurrence_count[expected_answer_id]
        answer_occurrence_count[expected_answer_id] += 1

        answer_positions = [
            i for i, tid in enumerate(input_ids) if tid == expected_answer_id
        ]

        assert target_occurrence < len(
            answer_positions
        ), f"Answer token {expected_answer_id} occurrence {target_occurrence} not found"

        answer_pos = answer_positions[target_occurrence]
        last_token_pos = answer_pos - 1

        assert last_token_pos >= 0, f"Answer token at position 0"
        positions.append((answer_pos, last_token_pos))

    return positions


def identify_chunks(prompt: str, tokenizer) -> dict[str, tuple[int, int]]:
    """
    Identify token spans for different input chunks.

    Following the paper, we identify:
    - init_token: "Init state:\n"
    - init_state: the actual initial state
    - goal_token: "Goal state:\n"
    - goal_state: the actual goal state
    - plan_token: "Plan:\n"
    - history_N: "step N: ..." for each completed step
    - action_prompt: the tokens right before the answer (e.g., "pick-up" or "on-top-of")

    Args:
        prompt: Full prompt string
        tokenizer: HuggingFace tokenizer

    Returns:
        Dict mapping chunk name to (start_pos, end_pos) token spans
    """
    # Tokenize the full prompt
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"][0].tolist()

    # Decode each token to find boundaries
    chunks = {}

    # Decode the full sequence to find character positions
    full_text = tokenizer.decode(input_ids)

    # Find character positions of key markers
    init_start_char = full_text.find("Init state:")
    goal_start_char = full_text.find("Goal state:")
    plan_start_char = full_text.find("Plan:")

    # Map character positions to token positions by decoding progressively
    def char_pos_to_token_pos(target_char_pos):
        """Binary search to find which token contains a character position"""
        for i in range(len(input_ids)):
            decoded_so_far = tokenizer.decode(input_ids[: i + 1])
            if len(decoded_so_far) >= target_char_pos:
                return i
        return len(input_ids) - 1

    # Identify chunk boundaries
    if init_start_char >= 0 and goal_start_char >= 0:
        init_token_start = char_pos_to_token_pos(init_start_char)
        goal_token_start = char_pos_to_token_pos(goal_start_char)
        chunks["init_state"] = (init_token_start, goal_token_start)

    if goal_start_char >= 0 and plan_start_char >= 0:
        goal_token_start = char_pos_to_token_pos(goal_start_char)
        plan_token_start = char_pos_to_token_pos(plan_start_char)
        chunks["goal_state"] = (goal_token_start, plan_token_start)

    # Find all step markers
    step_matches = list(re.finditer(r"step \d+:", full_text))
    for i, match in enumerate(step_matches):
        step_start_char = match.start()
        step_end_char = (
            step_matches[i + 1].start() if i + 1 < len(step_matches) else len(full_text)
        )

        step_token_start = char_pos_to_token_pos(step_start_char)
        step_token_end = char_pos_to_token_pos(step_end_char)
        chunks[f"history_{i+1}"] = (step_token_start, step_token_end)

    return chunks


def find_token_span(
    decoded_tokens: list[str], char_start: int, char_end: int
) -> tuple[int, int]:
    """
    Map character span to token span.

    Args:
        decoded_tokens: List of decoded token strings
        char_start: Start character position
        char_end: End character position

    Returns:
        (token_start, token_end) tuple
    """
    char_pos = 0
    token_start = -1
    token_end = -1

    for i, token in enumerate(decoded_tokens):
        if char_pos >= char_start and token_start == -1:
            token_start = i
        if char_pos >= char_end:
            token_end = i
            break
        char_pos += len(token)

    if token_end == -1:
        token_end = len(decoded_tokens)

    return (token_start, token_end)


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
def compute_information_flow(run_name: str, examples: list[dict], use_lora: bool):
    """
    Compute information flow from input chunks to last token across layers and steps.

    Returns a dictionary with:
        - results: nested dict mapping step -> layer -> chunk -> flow_score
        - num_layers: number of layers
        - num_examples: number of examples processed
    """
    import os
    import glob
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    import torch.nn.functional as F

    model_cache.reload()
    runs_volume.reload()

    os.environ["HF_HOME"] = "/cache"

    # Load model (same logic as extraction_rates.py)
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

    num_layers = model.config.num_hidden_layers

    # Results structure: {step_idx: {layer_idx: {chunk_name: flow_score}}}
    results = {}

    stats = {
        "examples_processed": 0,
        "steps_analyzed": 0,
    }

    print("Starting information flow computation...")

    for ex_idx, ex in enumerate(examples):
        # Construct full prompt with teacher-forcing
        prompt = f"""[INST] {ex['instruction']}

{ex['input']} [/INST]
{ex['output']}"""

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"][0].tolist()

        # Extract answer tokens and positions
        plan_text = ex["output"]
        expected_answers = extract_answer_tokens_from_plan(plan_text)
        expected_answer_ids = [
            tokenizer.encode(answer, add_special_tokens=False)[0]
            for answer in expected_answers
        ]
        positions = find_answer_token_positions(input_ids, expected_answer_ids)

        # Identify chunks
        chunks = identify_chunks(prompt, tokenizer)

        # Debug: Print identified chunks
        if ex_idx == 0:
            print(f"\nDEBUG: Identified {len(chunks)} chunks:")
            for chunk_name, (start, end) in chunks.items():
                print(f"  {chunk_name}: tokens [{start}:{end}]")

        stats["examples_processed"] += 1

        # For each step, compute information flow
        for step_idx, (answer_pos, last_token_pos) in enumerate(positions):
            stats["steps_analyzed"] += 1
            expected_answer_id = expected_answer_ids[step_idx]

            # Use baukit to trace attention weights during forward pass
            # This captures them while they're still in the computation graph
            from baukit import TraceDict

            # Trace all attention layers
            layer_names = [f"model.layers.{i}.self_attn" for i in range(num_layers)]

            with TraceDict(model, layer_names) as traces:
                # Forward pass - baukit captures intermediate values
                # CRITICAL: Must pass output_attentions=True so LlamaAttention returns attention weights
                outputs = model(**inputs, output_attentions=True, return_dict=True)

                # Retain gradients on captured attention weights
                # Note: We access the attention scores from the traced modules
                for layer_idx in range(num_layers):
                    layer_name = f"model.layers.{layer_idx}.self_attn"
                    # Get the attention weights from the trace
                    # The attention module outputs (attn_output, attn_weights, past_key_value)
                    # We need attn_weights which is the second element
                    if (
                        hasattr(traces[layer_name], "output")
                        and traces[layer_name].output is not None
                    ):
                        attn_output = traces[layer_name].output
                        # For LlamaAttention, output is (hidden_states, attn_weights, past_kv)
                        if isinstance(attn_output, tuple) and len(attn_output) >= 2:
                            attn_weights = attn_output[1]
                            if attn_weights is not None:
                                attn_weights.retain_grad()

            # Compute loss for this specific step
            logits = outputs.logits[0, last_token_pos, :]
            loss = F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([expected_answer_id]).to(model.device)
            )

            # Backward pass to get gradients
            model.zero_grad()
            loss.backward(retain_graph=True)

            # Compute information flow per layer
            step_key = f"step_{step_idx + 1}"
            if step_key not in results:
                results[step_key] = {}

            for layer_idx in range(num_layers):
                layer_name = f"model.layers.{layer_idx}.self_attn"

                # Get attention weights and gradients from traces
                if layer_name in traces:
                    attn_output = traces[layer_name].output
                    if isinstance(attn_output, tuple) and len(attn_output) >= 2:
                        attn_weights = attn_output[1]

                        if attn_weights is not None and attn_weights.grad is not None:
                            # Remove batch dimension: [batch, heads, seq, seq] -> [heads, seq, seq]
                            attention = attn_weights[0]
                            attention_grad = attn_weights.grad[0]

                            # Equation 7: I_token,ℓ(i,j) = |∑_hd A_hd,ℓ(j,i) ⊙ ∂L/∂A_hd,ℓ(j,i)|
                            # Element-wise multiply and sum over heads
                            flow = torch.abs(
                                (attention * attention_grad).sum(dim=0)
                            )  # [seq_len, seq_len]

                            # Extract flows FROM last_token_pos to all other tokens
                            # flow[i, j] = how much token i attends to token j
                            # We want: how much does last_token_pos attend to each token?
                            flow_from_last = (
                                flow[last_token_pos, :].detach().cpu().numpy()
                            )  # [seq_len]

                            # Aggregate to chunk level (Equation 8)
                            chunk_flows = {}
                            for chunk_name, (chunk_start, chunk_end) in chunks.items():
                                # Average flow from last token to this chunk
                                chunk_flow = flow_from_last[
                                    chunk_start:chunk_end
                                ].mean()
                                chunk_flows[chunk_name] = float(chunk_flow)

                            results[step_key][layer_idx] = chunk_flows
                        else:
                            print(
                                f"Warning: No attention weights or gradients for layer {layer_idx}"
                            )
                else:
                    print(f"Warning: Layer {layer_name} not traced")

            # Clean up for next step
            loss = None
            outputs = None
            torch.cuda.empty_cache()

        if (ex_idx + 1) % 5 == 0:
            print(f"Processed {ex_idx + 1}/{len(examples)} examples...")

    print(f"\n{'='*60}")
    print("Processing Statistics:")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print(f"{'='*60}\n")

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
    n_samples: int = 10,  # Start small for testing
    lora: bool = False,
):
    """
    Main entry point for information flow computation.
    """
    print(f"Loading test data from {data_file}...")
    with open(data_file, "r") as f:
        examples = [json.loads(line) for line in f]

    if n_samples > 0:
        examples = examples[:n_samples]
        print(f"Using first {n_samples} examples for testing.\n")
    else:
        print(f"Using all {len(examples)} examples.\n")

    model_type = "LoRA" if lora else "Full fine-tuned"
    print(f"Computing information flow on {model_type} model (run: {run_name})...")

    data = compute_information_flow.remote(run_name, examples, lora)

    print(f"\n{'='*60}")
    print(f"INFORMATION FLOW RESULTS ({model_type} Model)")
    print(f"{'='*60}")
    print(f"Run name: {run_name}")
    print(f"Dataset: {data_file}")
    print(f"Examples: {data['num_examples']}")
    print(f"Layers: {data['num_layers']}")
    print(f"{'='*60}\n")

    # Print sample results
    print("Sample results (Step 1, Layer 16):")
    if "step_1" in data["results"] and 16 in data["results"]["step_1"]:
        for chunk_name, flow_score in data["results"]["step_1"][16].items():
            print(f"  {chunk_name}: {flow_score:.4f}")

    # Save results
    model_name = run_name.replace("axo-", "").replace("/", "_")
    model_suffix = "lora" if lora else "full"
    output_file = f"results/information_flow_{model_suffix}_{model_name}.json"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nNext steps:")
    print("1. Generate heatmaps (Figure 5 style)")
    print("2. Compare goal_state vs history chunks")
    print("3. Analyze which chunks dominate at each layer")
