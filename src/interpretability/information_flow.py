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
    .apt_install("git")
    .pip_install(
        "transformers==4.44.0",
        "torch==2.3.0",
        "accelerate==0.33.0",
        "peft==0.12.0",
        "git+https://github.com/davidbau/baukit",
    )
)


def extract_answer_tokens_from_plan(plan_text: str) -> list[str]:
    """Extract the answer token (last word) from each step in the plan."""
    answers = []
    for line in plan_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("step"):
            tokens = line.split()
            if tokens:
                answers.append(tokens[-1])
    return answers


def char_pos_to_token_pos(input_ids: list[int], tokenizer, target_char_pos: int) -> int:
    """Find which token index contains a given character position."""
    for i in range(len(input_ids)):
        decoded_so_far = tokenizer.decode(input_ids[: i + 1])
        if len(decoded_so_far) >= target_char_pos:
            return i
    return len(input_ids) - 1


def find_answer_token_positions(
    input_ids: list[int],
    tokenizer,
    num_steps: int,
) -> list[tuple[int, int]]:
    """
    Find positions of answer tokens by locating the end of each step line.

    Returns:
        List of (answer_pos, last_token_pos) tuples for each step.
    """
    full_text = tokenizer.decode(input_ids)
    positions = []

    for step_num in range(1, num_steps + 1):
        step_pattern = f"step {step_num}:"
        step_start = full_text.find(step_pattern)
        assert step_start >= 0, f"Could not find '{step_pattern}' in prompt"

        next_step_pattern = f"step {step_num + 1}:"
        step_end = full_text.find(next_step_pattern)
        if step_end == -1:
            step_end = len(full_text)

        step_line_end_pos = char_pos_to_token_pos(input_ids, tokenizer, step_end)

        # Find the last non-whitespace token before step line ends
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

    Chunks identified (matching paper Figure 5):
    - init_token: "Init state:"
    - init_state: the actual initial state content
    - goal_token: "Goal state:"
    - goal_state: the actual goal state content
    - plan_token: "Plan:"
    - history_N: completed step N
    """
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"][0].tolist()
    full_text = tokenizer.decode(input_ids)
    chunks = {}

    def to_token_pos(char_pos: int) -> int:
        return char_pos_to_token_pos(input_ids, tokenizer, char_pos)

    # Find key markers
    init_start = full_text.find("Init state:")
    goal_start = full_text.find("Goal state:")
    plan_start = full_text.find("Plan:")

    # Init token and state
    if init_start >= 0 and goal_start >= 0:
        init_token_start = to_token_pos(init_start)
        state_content_start = full_text.find("<", init_start)

        if init_start < state_content_start < goal_start:
            chunks["init_token"] = (init_token_start, to_token_pos(state_content_start))
            chunks["init_state"] = (
                to_token_pos(state_content_start),
                to_token_pos(goal_start),
            )
        else:
            chunks["init_state"] = (init_token_start, to_token_pos(goal_start))

    # Goal token and state
    if goal_start >= 0 and plan_start >= 0:
        goal_token_start = to_token_pos(goal_start)
        state_content_start = full_text.find("<", goal_start)

        if goal_start < state_content_start < plan_start:
            chunks["goal_token"] = (goal_token_start, to_token_pos(state_content_start))
            chunks["goal_state"] = (
                to_token_pos(state_content_start),
                to_token_pos(plan_start),
            )
        else:
            chunks["goal_state"] = (goal_token_start, to_token_pos(plan_start))

        # Plan token
        first_step = re.search(r"step 1:", full_text)
        if first_step:
            chunks["plan_token"] = (
                to_token_pos(plan_start),
                to_token_pos(first_step.start()),
            )

    # History steps
    for i, match in enumerate(re.finditer(r"step \d+:", full_text)):
        step_start = match.start()
        next_match = re.search(rf"step {i + 2}:", full_text)
        step_end = next_match.start() if next_match else len(full_text)
        chunks[f"history_{i + 1}"] = (to_token_pos(step_start), to_token_pos(step_end))

    return chunks


@app.function(
    image=hf_image,
    gpu="a100-80gb",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": model_cache, "/runs": runs_volume},
    timeout=10800,
)
def compute_information_flow(run_name: str, examples: list[dict], use_lora: bool):
    """Compute information flow from input chunks to last token across layers and steps."""
    import os
    import glob
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    import torch.nn.functional as F
    from baukit import TraceDict

    model_cache.reload()
    runs_volume.reload()
    os.environ["HF_HOME"] = "/cache"

    # Load model
    if use_lora:
        base_model_id = "meta-llama/Llama-2-7b-chat-hf"
        lora_path = f"/runs/{run_name}/lora-out"
        print(f"Loading base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="auto"
        )
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
    else:
        model_path = f"/runs/{run_name}/full-out"
        checkpoints = sorted(glob.glob(f"{model_path}/checkpoint-*"))
        if checkpoints:
            model_path = checkpoints[-1]
            print(f"Loading from checkpoint: {model_path}")
        print(f"Loading full fine-tuned model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(f"/runs/{run_name}/full-out")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

    model.eval()
    model_cache.commit()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded! {len(examples)} examples, {num_layers} layers")

    accumulated_results = {}
    step_counts = {}
    stats = {"examples_processed": 0, "steps_analyzed": 0}

    for ex_idx, ex in enumerate(examples):
        prompt = f"[INST] {ex['instruction']}\n\n{ex['input']} [/INST]\n{ex['output']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"][0].tolist()

        expected_answers = extract_answer_tokens_from_plan(ex["output"])
        expected_answer_ids = [
            tokenizer.encode(ans, add_special_tokens=False)[0]
            for ans in expected_answers
        ]
        positions = find_answer_token_positions(
            input_ids, tokenizer, len(expected_answers)
        )
        chunks = identify_chunks(prompt, tokenizer)

        if ex_idx == 0:
            print(f"\nDEBUG: Identified {len(chunks)} chunks:")
            for name, (start, end) in chunks.items():
                print(
                    f"  {name}: [{start}:{end}] {tokenizer.decode(input_ids[start:end])}"
                )

        stats["examples_processed"] += 1

        for step_idx, (answer_pos, last_token_pos) in enumerate(positions):
            stats["steps_analyzed"] += 1
            expected_answer_id = expected_answer_ids[step_idx]

            layer_names = [f"model.layers.{i}.self_attn" for i in range(num_layers)]

            with TraceDict(model, layer_names) as traces:
                outputs = model(**inputs, output_attentions=True, return_dict=True)
                for layer_idx in range(num_layers):
                    layer_name = f"model.layers.{layer_idx}.self_attn"
                    attn_output = getattr(traces[layer_name], "output", None)
                    if isinstance(attn_output, tuple) and len(attn_output) >= 2:
                        attn_weights = attn_output[1]
                        if attn_weights is not None:
                            attn_weights.retain_grad()

            logits = outputs.logits[0, last_token_pos, :]
            loss = F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([expected_answer_id]).to(model.device)
            )
            model.zero_grad()
            loss.backward(retain_graph=True)

            step_key = f"step_{step_idx + 1}"
            if step_key not in accumulated_results:
                accumulated_results[step_key] = {}
                step_counts[step_key] = 0
            step_counts[step_key] += 1

            for layer_idx in range(num_layers):
                layer_name = f"model.layers.{layer_idx}.self_attn"
                attn_output = getattr(traces[layer_name], "output", None)

                if not isinstance(attn_output, tuple) or len(attn_output) < 2:
                    continue

                attn_weights = attn_output[1]
                if attn_weights is None or attn_weights.grad is None:
                    continue

                attention = attn_weights[0]
                attention_grad = attn_weights.grad[0]

                # Equation 7: I_token,ℓ(i,j) = |∑_hd A_hd,ℓ(j,i) ⊙ ∂L/∂A_hd,ℓ(j,i)|
                flow = torch.abs((attention * attention_grad).sum(dim=0))
                flow_from_last = flow[last_token_pos, :].detach().cpu().numpy()

                # Aggregate to chunk level (Equation 8)
                for chunk_name, (chunk_start, chunk_end) in chunks.items():
                    # Rename current step to "action_prompt"
                    display_name = (
                        "action_prompt"
                        if chunk_name == f"history_{step_idx + 1}"
                        else chunk_name
                    )
                    chunk_flow = float(flow_from_last[chunk_start:chunk_end].mean())

                    if layer_idx not in accumulated_results[step_key]:
                        accumulated_results[step_key][layer_idx] = {}
                    if display_name not in accumulated_results[step_key][layer_idx]:
                        accumulated_results[step_key][layer_idx][display_name] = 0.0
                    accumulated_results[step_key][layer_idx][display_name] += chunk_flow

            torch.cuda.empty_cache()

        if (ex_idx + 1) % 5 == 0:
            print(f"Processed {ex_idx + 1}/{len(examples)} examples...")

    # Average results
    results = {}
    for step_key, step_data in accumulated_results.items():
        results[step_key] = {}
        count = step_counts[step_key]
        for layer_idx, layer_data in step_data.items():
            results[step_key][layer_idx] = {
                chunk: flow / count for chunk, flow in layer_data.items()
            }

    print(
        f"\nProcessed {stats['examples_processed']} examples, {stats['steps_analyzed']} steps"
    )

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
    """Main entry point for information flow computation."""
    print(f"Loading test data from {data_file}...")
    with open(data_file, "r") as f:
        examples = [json.loads(line) for line in f]

    if n_samples > 0:
        examples = examples[:n_samples]
        print(f"Using first {n_samples} examples.\n")
    else:
        print(f"Using all {len(examples)} examples.\n")

    model_type = "LoRA" if lora else "Full fine-tuned"
    print(f"Computing information flow on {model_type} model (run: {run_name})...")

    data = compute_information_flow.remote(run_name, examples, lora)

    print(f"\n{'='*60}")
    print(f"INFORMATION FLOW RESULTS ({model_type} Model)")
    print(f"{'='*60}")
    print(f"Run: {run_name} | Dataset: {data_file}")
    print(f"Examples: {data['num_examples']} | Layers: {data['num_layers']}")
    print(f"{'='*60}\n")

    # Print sample results
    sample_layer = 16
    print(f"Sample results at Layer {sample_layer}:\n")
    for step in ["step_1", "step_2", "step_3", "step_4", "step_5"]:
        if step in data["results"] and sample_layer in data["results"][step]:
            print(f"{step.replace('_', ' ').title()}:")
            for chunk, score in data["results"][step][sample_layer].items():
                print(f"  {chunk}: {score:.4f}")
            print()

    # Save results
    model_name = run_name.replace("axo-", "").replace("/", "_")
    model_suffix = "lora" if lora else "full"
    output_file = f"results/information_flow_{model_suffix}_{model_name}.json"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output_file}")
