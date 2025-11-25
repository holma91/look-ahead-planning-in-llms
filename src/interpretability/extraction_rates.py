"""
Compute extraction rates for MLP and MHSA components across layers.

This implements Section 5.1 from the paper: "Attention Extract the Answers"

The extraction rate measures whether a component's hidden state at the last token
can directly decode the correct answer without going through the rest of the model.

Usage:
    # Run on full fine-tuned model
    uv run modal run src.interpretability.extraction_rates \
        --run-name axo-2025-11-17-12-42-41-3911 \
        --data-file data/blocksworld_L3_correct_full.jsonl

    # Run on LoRA model
    uv run modal run src.interpretability.extraction_rates \
        --run-name axo-2025-11-17-09-14-09-9bf6 \
        --data-file data/blocksworld_L3_correct_lora.jsonl \
        --lora
"""

import modal
import json

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
    """Extract the answer token (last word) from each step in the plan."""
    return [
        line.split()[-1]
        for line in plan_text.strip().split("\n")
        if line.strip().startswith("step") and line.split()
    ]


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
    expected_answer_ids: list[int],
) -> list[tuple[int, int]]:
    """
    Find positions of answer tokens by scanning the plan section sequentially.

    We start from the first occurrence of "step 1:" and, for each expected answer
    token ID, locate the next matching token in the plan. This avoids matching
    colors that appear earlier in the prompt (e.g., in the init/goal states).
    """
    full_text = tokenizer.decode(input_ids)
    first_step_char = full_text.find("step 1:")
    assert first_step_char >= 0, "Could not find 'step 1:' in prompt"

    search_idx = char_pos_to_token_pos(input_ids, tokenizer, first_step_char)
    positions = []

    for step_idx, expected_id in enumerate(expected_answer_ids):
        found_pos = None
        for idx in range(search_idx, len(input_ids)):
            if input_ids[idx] == expected_id:
                found_pos = idx
                search_idx = idx + 1
                break
        if found_pos is None:
            raise ValueError(
                f"Could not find token ID {expected_id} for step {step_idx + 1}"
            )

        answer_pos = found_pos
        last_token_pos = answer_pos - 1
        assert last_token_pos >= 0, "Answer token cannot be at position 0"
        positions.append((answer_pos, last_token_pos))

    return positions


def check_extraction_event(hidden_state, lm_head, expected_token_id: int) -> bool:
    """Check if a hidden state can directly decode to the expected token."""
    import torch

    logits = torch.matmul(hidden_state, lm_head.T)
    return torch.argmax(logits).item() == expected_token_id


class ComponentOutputCapture:
    """Captures MHSA and MLP outputs from each layer using forward hooks."""

    def __init__(self, model, num_layers: int):
        self.mhsa_outputs = {}
        self.mlp_outputs = {}
        self.hooks = []

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            self.hooks.append(
                layer.self_attn.register_forward_hook(
                    self._make_hook(layer_idx, "mhsa")
                )
            )
            self.hooks.append(
                layer.mlp.register_forward_hook(self._make_hook(layer_idx, "mlp"))
            )

    def _make_hook(self, layer_idx: int, component: str):
        def hook(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            storage = self.mhsa_outputs if component == "mhsa" else self.mlp_outputs
            storage[layer_idx] = tensor.detach().cpu()

        return hook

    def clear(self):
        self.mhsa_outputs.clear()
        self.mlp_outputs.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


@app.function(
    image=hf_image,
    gpu="a100-80gb",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": model_cache, "/runs": runs_volume},
    timeout=10800,
)
def compute_extraction_rates(run_name: str, examples: list[dict], use_lora: bool):
    """Compute extraction rates for MLP, MHSA, and layer outputs across all layers."""
    import os
    import glob
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    model_cache.reload()
    runs_volume.reload()
    os.environ["HF_HOME"] = "/cache"

    # Load model
    if use_lora:
        base_model_id = "meta-llama/Llama-2-7b-chat-hf"
        lora_path = f"/runs/{run_name}/lora-out"
        print(f"Loading LoRA model from: {lora_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, lora_path).merge_and_unload()
    else:
        model_path = f"/runs/{run_name}/full-out"
        checkpoints = sorted(glob.glob(f"{model_path}/checkpoint-*"))
        if checkpoints:
            model_path = checkpoints[-1]
        print(f"Loading full fine-tuned model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(f"/runs/{run_name}/full-out")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

    model.eval()
    model_cache.commit()

    num_layers = model.config.num_hidden_layers
    lm_head = model.lm_head.weight
    print(f"Model loaded! {len(examples)} examples, {num_layers} layers\n")

    # Track extraction events per layer AND per step (for variance calculation)
    # Structure: {component: {step_idx: [[] for each layer]}}
    extraction_events_by_step = {
        comp: {step: [[] for _ in range(num_layers)] for step in range(6)}
        for comp in ["mlp", "mhsa", "layer"]
    }
    stats = {
        "examples_processed": 0,
        "steps_analyzed": 0,
        "steps_with_correct_final_prediction": 0,
    }

    component_capture = ComponentOutputCapture(model, num_layers)

    for ex_idx, ex in enumerate(examples):
        component_capture.clear()

        prompt = f"[INST] {ex['instruction']}\n\n{ex['input']} [/INST]\n{ex['output']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states
        logits = outputs.logits
        input_ids = inputs["input_ids"][0].tolist()

        expected_answers = extract_answer_tokens_from_plan(ex["output"])
        expected_answer_ids = [
            tokenizer.encode(ans, add_special_tokens=False)[0]
            for ans in expected_answers
        ]
        positions = find_answer_token_positions(
            input_ids, tokenizer, expected_answer_ids
        )
        stats["examples_processed"] += 1

        # Debug logging for first 2 examples
        if ex_idx < 2:
            print(f"\n{'='*60}")
            print(f"Example {ex_idx}: {len(input_ids)} tokens")
            print(f"Expected answers: {expected_answers}")
            print(f"Expected answer IDs: {expected_answer_ids}")
            print(f"Positions (answer_pos, last_token_pos): {positions}")
            print(f"Output text:\n{ex['output']}")

        for step_idx, (answer_pos, last_token_pos) in enumerate(positions):
            stats["steps_analyzed"] += 1
            expected_id = expected_answer_ids[step_idx]
            predicted_id = torch.argmax(logits[0, last_token_pos, :]).item()

            # Debug logging for first 2 examples
            if ex_idx < 2:
                expected_token = tokenizer.decode([expected_id])
                predicted_token = tokenizer.decode([predicted_id])
                context_tokens = tokenizer.decode(
                    input_ids[max(0, last_token_pos - 5) : last_token_pos + 1]
                )
                print(
                    f"  Step {step_idx+1}: last_token_pos={last_token_pos}, answer_pos={answer_pos}"
                )
                print(f"    Context: '...{context_tokens}'")
                print(f"    Expected: '{expected_token}' (id={expected_id})")
                print(f"    Predicted: '{predicted_token}' (id={predicted_id})")
                print(f"    Correct: {predicted_id == expected_id}")

            if predicted_id != expected_id:
                continue

            stats["steps_with_correct_final_prediction"] += 1

            for layer_idx in range(num_layers):
                h_layer = hidden_states[layer_idx + 1][0, last_token_pos, :]
                extraction_events_by_step["layer"][step_idx][layer_idx].append(
                    check_extraction_event(h_layer, lm_head, expected_id)
                )

                h_mhsa = component_capture.mhsa_outputs[layer_idx][
                    0, last_token_pos, :
                ].to(model.device)
                extraction_events_by_step["mhsa"][step_idx][layer_idx].append(
                    check_extraction_event(h_mhsa, lm_head, expected_id)
                )

                h_mlp = component_capture.mlp_outputs[layer_idx][
                    0, last_token_pos, :
                ].to(model.device)
                extraction_events_by_step["mlp"][step_idx][layer_idx].append(
                    check_extraction_event(h_mlp, lm_head, expected_id)
                )

        if (ex_idx + 1) % 10 == 0:
            print(f"Processed {ex_idx + 1}/{len(examples)} examples...")

    component_capture.remove_hooks()

    print(f"\nStats: {stats}")

    # Compute per-step extraction rates for each layer
    # Structure: {component: {step_idx: [rate_layer_0, rate_layer_1, ...]}}
    per_step_rates = {}
    for comp in ["mlp", "mhsa", "layer"]:
        per_step_rates[comp] = {}
        for step_idx in range(6):
            per_step_rates[comp][step_idx] = [
                sum(events) / len(events) if events else 0.0
                for events in extraction_events_by_step[comp][step_idx]
            ]

    # Compute mean and std across steps for each layer
    import numpy as np

    results = {}
    results_std = {}
    for comp in ["mlp", "mhsa", "layer"]:
        # Stack per-step rates: shape (6 steps, num_layers)
        step_rates_array = np.array([per_step_rates[comp][step] for step in range(6)])
        results[comp] = step_rates_array.mean(axis=0).tolist()
        results_std[comp] = step_rates_array.std(axis=0).tolist()

    return {
        "results": results,
        "results_std": results_std,
        "per_step_rates": per_step_rates,
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
    print(f"Loading test data from {data_file}...")
    with open(data_file, "r") as f:
        examples = [json.loads(line) for line in f]

    if n_samples > 0:
        examples = examples[:n_samples]
    print(
        f"Using {'first ' + str(n_samples) if n_samples else 'all ' + str(len(examples))} examples.\n"
    )

    model_type = "LoRA" if lora else "Full fine-tuned"
    print(f"Computing extraction rates on {model_type} model (run: {run_name})...")

    data = compute_extraction_rates.remote(run_name, examples, lora)

    print(f"\nEXTRACTION RATE RESULTS ({model_type} Model)")
    print(
        f"Run: {run_name}, Examples: {data['num_examples']}, Layers: {data['num_layers']}\n"
    )

    print(f"{'Layer':<8} {'MHSA':<12} {'MLP':<12} {'Layer Output':<15}")
    print("-" * 50)
    for i in range(data["num_layers"]):
        print(
            f"{i:<8} {data['results']['mhsa'][i]:.3f}        "
            f"{data['results']['mlp'][i]:.3f}        {data['results']['layer'][i]:.3f}"
        )

    model_name = run_name.replace("axo-", "").replace("/", "_")
    output_file = (
        f"results/extraction_rates_{'lora' if lora else 'full'}_{model_name}.json"
    )
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_file}")
