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
    tokenizer,
    num_steps: int,
) -> list[tuple[int, int]]:
    """
    Find the positions of answer tokens by locating the end of each step line.
    Answer is always the LAST token on each "step N:" line.

    Args:
        input_ids: Token IDs from tokenizer
        tokenizer: HuggingFace tokenizer
        num_steps: Number of steps in the plan

    Returns:
        List of (answer_pos, last_token_pos) tuples for each step.
    """
    full_text = tokenizer.decode(input_ids)
    positions = []

    # Find each "step N:" line
    for step_num in range(1, num_steps + 1):
        # Find the start and end of this step's line
        step_pattern = f"step {step_num}:"
        step_start = full_text.find(step_pattern)
        assert step_start >= 0, f"Could not find '{step_pattern}' in prompt"

        # Find where this step's line ends (either next step or end of text)
        next_step_pattern = f"step {step_num + 1}:"
        step_end = full_text.find(next_step_pattern)
        if step_end == -1:
            step_end = len(full_text)

        # Map character positions to token positions
        step_line_end_pos = None
        for i in range(len(input_ids)):
            decoded = tokenizer.decode(input_ids[: i + 1])
            if len(decoded) >= step_end:
                step_line_end_pos = i
                break

        assert (
            step_line_end_pos is not None
        ), f"Could not find token position for step {step_num} end"

        # The answer token is right before the step line ends
        # Go back to find the last non-newline token
        answer_pos = step_line_end_pos - 1
        while answer_pos > 0:
            token_text = tokenizer.decode([input_ids[answer_pos]])
            if token_text.strip() and token_text not in ["\n", " "]:
                break
            answer_pos -= 1

        last_token_pos = answer_pos - 1
        assert last_token_pos >= 0, f"Answer token at position 0 for step {step_num}"

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

    # Results structure: {step_idx: {layer_idx: {chunk_name: accumulated_flow_score}}}
    # We'll accumulate flows across all examples, then average at the end
    accumulated_results = {}
    step_counts = {}  # Track how many examples contributed to each step

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
        num_steps = len(expected_answers)
        print(f"Expected answers: {expected_answers}")

        # Get token IDs for each answer (needed for loss computation)
        expected_answer_ids = [
            tokenizer.encode(answer, add_special_tokens=False)[0]
            for answer in expected_answers
        ]
        print(f"Expected answer IDs: {expected_answer_ids}")

        # Find positions by locating the end of each step line
        positions = find_answer_token_positions(input_ids, tokenizer, num_steps)
        print(f"Positions: {positions}")
        chunks = identify_chunks(prompt, tokenizer)
        if ex_idx == 0:
            print(f"\nDEBUG: Identified {len(chunks)} chunks:")
            for chunk_name, (start, end) in chunks.items():
                print(f"  {chunk_name}: tokens [{start}:{end}]")
                print(f"  {chunk_name}: text {tokenizer.decode(input_ids[start:end])}")

        stats["examples_processed"] += 1

        # For each step, compute information flow
        for step_idx, (answer_pos, last_token_pos) in enumerate(positions):
            # e.g 6, (141, 140) where expected_answer_id = 7933 which is the token for "green"
            # -> "what is the model attending to at position 140 when predicting green at position 141?"
            stats["steps_analyzed"] += 1
            expected_answer_id = expected_answer_ids[step_idx]

            # we use baukit to trace attention weights during forward pass
            # baukit.TraceDict intercepts and saves the attention weights at each layer while the forward pass is running
            from baukit import TraceDict

            layer_names = [f"model.layers.{i}.self_attn" for i in range(num_layers)]
            with TraceDict(model, layer_names) as traces:
                # critical: must pass output_attentions=True so LlamaAttention returns attention weights
                outputs = model(**inputs, output_attentions=True, return_dict=True)
                for layer_idx in range(num_layers):
                    layer_name = f"model.layers.{layer_idx}.self_attn"
                    if (
                        hasattr(traces[layer_name], "output")
                        and traces[layer_name].output is not None
                    ):
                        attn_output = traces[layer_name].output
                        # For LlamaAttention, output is (hidden_states, attn_weights, past_kv)
                        if isinstance(attn_output, tuple) and len(attn_output) >= 2:
                            attn_weights = attn_output[1]
                            if attn_weights is not None:
                                # attn_weights shape: [batch, heads, seq_len, seq_len]
                                seq_len = len(input_ids)
                                assert attn_weights.shape == (
                                    1,
                                    32,
                                    seq_len,
                                    seq_len,
                                ), f"Expected shape (1, 32, {seq_len}, {seq_len}), got {attn_weights.shape}"
                                attn_weights.retain_grad()

            logits = outputs.logits[0, last_token_pos, :]  # get logits at e.g pos 140
            loss = F.cross_entropy(  # how wrong is the model at predicting 7933 (green) at position 141?
                logits.unsqueeze(0), torch.tensor([expected_answer_id]).to(model.device)
            )

            # backward pass to get the gradients
            # such that, for each layer,attn_weights.grad contains the gradients of the loss with respect to the attention weights
            model.zero_grad()
            loss.backward(retain_graph=True)

            step_key = f"step_{step_idx + 1}"
            if step_key not in accumulated_results:
                accumulated_results[step_key] = {}
                step_counts[step_key] = 0
            step_counts[step_key] += 1

            # compute information flow per layer
            for layer_idx in range(num_layers):
                layer_name = f"model.layers.{layer_idx}.self_attn"

                if layer_name in traces:
                    attn_output = traces[layer_name].output
                    if isinstance(attn_output, tuple) and len(attn_output) >= 2:
                        attn_weights = attn_output[1]  # [batch, heads, seq, seq]

                        if attn_weights is not None and attn_weights.grad is not None:
                            attention = attn_weights[0]  # [heads, seq, seq]
                            attention_grad = attn_weights.grad[  # ∂L/∂A for this layer
                                0
                            ]

                            # Equation 7 from the paper: I_token,ℓ(i,j) = |∑_hd A_hd,ℓ(j,i) ⊙ ∂L/∂A_hd,ℓ(j,i)|
                            flow = torch.abs((attention * attention_grad).sum(dim=0))

                            # Extract flows FROM last_token_pos to all other tokens
                            # flow[i, j] = how much token i attends to token j
                            # We want: how much does last_token_pos attend to each token?
                            flow_from_last = (
                                flow[last_token_pos, :].detach().cpu().numpy()
                            )

                            # aggregating to chunk level (Equation 8)
                            # for each chunk, how much flows from the chunk to the last token?
                            # e.g 0.005 could flow from the goal state chunk to the last token
                            for chunk_name, (chunk_start, chunk_end) in chunks.items():
                                # average flow from last token to this chunk
                                chunk_flow = flow_from_last[
                                    chunk_start:chunk_end
                                ].mean()

                                # Initialize nested dicts if needed
                                if layer_idx not in accumulated_results[step_key]:
                                    accumulated_results[step_key][layer_idx] = {}
                                if (
                                    chunk_name
                                    not in accumulated_results[step_key][layer_idx]
                                ):
                                    accumulated_results[step_key][layer_idx][
                                        chunk_name
                                    ] = 0.0

                                # Accumulate flow scores
                                accumulated_results[step_key][layer_idx][
                                    chunk_name
                                ] += float(chunk_flow)

            loss = None
            outputs = None
            torch.cuda.empty_cache()

        if (ex_idx + 1) % 5 == 0:
            print(f"Processed {ex_idx + 1}/{len(examples)} examples...")

    # Average the accumulated results across examples
    print(f"\nAveraging results across {len(examples)} examples...")
    results = {}
    for step_key, step_data in accumulated_results.items():
        results[step_key] = {}
        count = step_counts[step_key]
        for layer_idx, layer_data in step_data.items():
            results[step_key][layer_idx] = {}
            for chunk_name, accumulated_flow in layer_data.items():
                results[step_key][layer_idx][chunk_name] = accumulated_flow / count

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
    n_samples: int = 0,  # 0 = all examples (default, like the paper)
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

    # Print sample results for multiple steps
    sample_steps = ["step_1", "step_2", "step_3", "step_4", "step_5"]
    sample_layer = 16

    print(f"Sample results at Layer {sample_layer}:\n")
    for step in sample_steps:
        if step in data["results"] and sample_layer in data["results"][step]:
            print(f"{step.replace('_', ' ').title()}:")
            for chunk_name, flow_score in data["results"][step][sample_layer].items():
                print(f"  {chunk_name}: {flow_score:.4f}")
            print()

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
