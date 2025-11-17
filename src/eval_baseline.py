"""
Baseline evaluation script for Llama-2-7b-chat-hf (instruction-tuned, no fine-tuning on Blocksworld).
Runs inference on Blocksworld test data to establish baseline metrics before task-specific fine-tuning.
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("blocksworld-baseline-eval")

# Cache for HuggingFace models
model_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

hf_image = modal.Image.debian_slim().pip_install(
    "transformers==4.44.0",
    "torch==2.3.0",
    "accelerate==0.33.0",
)


@app.function(
    image=hf_image,
    gpu="a10g",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": model_cache},
    timeout=1800,  # 30 minutes for batch processing
)
def generate_batch(examples: list[dict]) -> list[dict]:
    """Generate predictions for a batch of examples."""
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_cache.reload()
    os.environ["HF_HOME"] = "/cache"

    model_id = (
        "meta-llama/Llama-2-7b-chat-hf"  # Instruction-tuned version (same as paper)
    )

    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_cache.commit()
    print(f"Model loaded! Processing {len(examples)} examples...")

    results = []
    for i, example in enumerate(examples):
        print(f"Processing example {i+1}/{len(examples)}...")

        # Format prompt (same as fine-tuning format)
        prompt = f"{example['instruction']}\n\n{example['input']}\n\nPlan:"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Longer for multi-step plans
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from prediction
        prediction = prediction[len(prompt) :].strip()

        results.append(
            {
                "example_id": i,
                "input": example["input"],
                "expected": example["output"],
                "predicted": prediction,
            }
        )

    return results


def load_test_data(filepath: str, n_samples: int = None) -> list[dict]:
    """Load test examples from JSONL file."""
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            examples.append(json.loads(line))
            if n_samples and len(examples) >= n_samples:
                break
    return examples


def calculate_metrics(results: list[dict]) -> dict:
    """Calculate evaluation metrics."""
    exact_matches = 0
    has_plan_prefix = 0
    has_steps = 0

    for result in results:
        expected = result["expected"].strip()
        predicted = result["predicted"].strip()

        # Exact match
        if expected == predicted:
            exact_matches += 1

        # Has "Plan:" or "step" in output (format correctness)
        if "Plan:" in predicted or "step" in predicted.lower():
            has_plan_prefix += 1

        # Has numbered steps
        if "step 1:" in predicted.lower():
            has_steps += 1

    n = len(results)
    return {
        "total_examples": n,
        "exact_match": exact_matches / n if n > 0 else 0,
        "format_correctness": has_plan_prefix / n if n > 0 else 0,
        "has_steps": has_steps / n if n > 0 else 0,
    }


@app.local_entrypoint()
def main(
    data_file: str = "data/blocksworld_L3.jsonl",
    n_samples: int = 20,
    output_file: str = "results/baseline_llama2_7b_chat.json",
):
    """
    Run baseline evaluation.

    Args:
        data_file: Path to test data JSONL file
        n_samples: Number of samples to evaluate
        output_file: Where to save results
    """
    print(f"Loading {n_samples} examples from {data_file}...")
    examples = load_test_data(data_file, n_samples)
    print(f"Loaded {len(examples)} examples")

    print(f"\nRunning inference on Llama-2-7b-chat-hf (pre-fine-tuning baseline)...")
    results = generate_batch.remote(examples)

    print(f"\nCalculating metrics...")
    metrics = calculate_metrics(results)

    print("\n" + "=" * 60)
    print("BASELINE METRICS")
    print("=" * 60)
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Exact match: {metrics['exact_match']:.1%}")
    print(f"Format correctness: {metrics['format_correctness']:.1%}")
    print(f"Has steps: {metrics['has_steps']:.1%}")
    print("=" * 60)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    output_data = {"metrics": metrics, "results": results}

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"\nSample predictions:")
    for i, result in enumerate(results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {result['input'][:100]}...")
        print(f"Expected: {result['expected'][:100]}...")
        print(f"Predicted: {result['predicted'][:100]}...")
