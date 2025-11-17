"""
Evaluate fine-tuned LoRA model on Blocksworld test set.
Compares performance against the baseline (0% accuracy).

Usage:
    # Evaluate on L3 test set (to compare with paper's 61% result)
    uv run modal run src.eval_finetuned --run-name axo-2025-11-17-08-26-35-b9fa \
        --data-file data/blocksworld_L3_test.jsonl --n-samples 100

    # Evaluate on full test set
    uv run modal run src.eval_finetuned --run-name axo-2025-11-17-08-26-35-b9fa \
        --data-file data/blocksworld_test.jsonl --n-samples 200
"""

import modal
import json

app = modal.App("eval-finetuned-blocksworld")

model_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("example-runs-vol", create_if_missing=True)

hf_image = modal.Image.debian_slim().pip_install(
    "transformers==4.44.0",
    "torch==2.3.0",
    "accelerate==0.33.0",
    "peft==0.12.0",
)


@app.function(
    image=hf_image,
    gpu="a10g",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": model_cache,
        "/runs": runs_volume,
    },
    timeout=1800,
)
def generate_batch(run_name: str, examples: list[dict]) -> list[dict]:
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    model_cache.reload()
    runs_volume.reload()

    os.environ["HF_HOME"] = "/cache"

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
    model.eval()

    model_cache.commit()

    print(f"Model loaded! Processing {len(examples)} examples...")

    results = []
    for i, ex in enumerate(examples):
        prompt = f"""[INST] {ex['instruction']}

{ex['input']} [/INST]"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = generated[len(prompt) :].strip()

        results.append(
            {
                "example_id": i,
                "input": ex["input"],
                "expected": ex["output"],
                "predicted": prediction,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples...")

    return results


@app.local_entrypoint()
def main(
    run_name: str = "axo-2025-11-17-08-26-35-b9fa",
    data_file: str = "data/blocksworld_L3_test.jsonl",
    n_samples: int = 50,
    output_file: str = "",
):
    print(f"Loading test data from {data_file}...")
    with open(data_file, "r") as f:
        examples = [json.loads(line) for line in f][:n_samples]

    print(f"Loaded {len(examples)} examples for evaluation.\n")

    print(f"Running inference on fine-tuned model (run: {run_name})...")
    results = generate_batch.remote(run_name, examples)

    exact_match = sum(
        1 for r in results if r["predicted"].strip() == r["expected"].strip()
    ) / len(results)

    has_plan_keyword = sum(
        1 for r in results if "plan:" in r["predicted"].lower()
    ) / len(results)

    has_steps = sum(1 for r in results if "step" in r["predicted"].lower()) / len(
        results
    )

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (Fine-tuned Model)")
    print(f"{'='*60}")
    print(f"Run name: {run_name}")
    print(f"Dataset: {data_file}")
    print(f"Samples: {len(results)}")
    print(f"\nMetrics:")
    print(f"  Exact Match:       {exact_match:.2%}")
    print(f"  Has 'Plan:' :      {has_plan_keyword:.2%}")
    print(f"  Has 'step':        {has_steps:.2%}")
    print(f"{'='*60}\n")

    if not output_file:
        model_name = run_name.replace("axo-", "").replace("/", "_")
        output_file = f"results/finetuned_{model_name}.json"

    output_data = {
        "run_name": run_name,
        "metrics": {
            "total_examples": len(results),
            "exact_match": exact_match,
            "has_plan_keyword": has_plan_keyword,
            "has_steps": has_steps,
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")

    print(f"\n{'='*60}")
    print("Sample predictions:")
    print(f"{'='*60}\n")
    for i, r in enumerate(results[:3]):
        print(f"Example {i+1}:")
        print(f"Expected:\n{r['expected']}\n")
        print(f"Predicted:\n{r['predicted']}\n")
        print(f"{'-'*60}\n")
