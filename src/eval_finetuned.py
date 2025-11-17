"""
Evaluate fine-tuned models (LoRA or full) on Blocksworld test set.

Usage:
    # Evaluate LoRA model on L3 test set
    uv run modal run src.eval_finetuned --run-name axo-2025-11-17-09-14-09-9bf6 \
        --data-file data/blocksworld_L3_test.jsonl --n-samples 50 --lora

    # Evaluate full fine-tuned model
    uv run modal run src.eval_finetuned --run-name axo-2025-11-17-09-20-23-e552 \
        --data-file data/blocksworld_L3_test.jsonl --n-samples 50
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
    timeout=7200,
)
def generate_batch(run_name: str, examples: list[dict], use_lora: bool) -> list[dict]:
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
    run_name: str = "axo-2025-11-17-09-20-23-e552",
    data_file: str = "data/blocksworld_L3_test.jsonl",
    n_samples: int = 0,
    lora: bool = False,
    output_file: str = "",
):
    print(f"Loading test data from {data_file}...")
    with open(data_file, "r") as f:
        examples = [json.loads(line) for line in f]

    if n_samples > 0:
        examples = examples[:n_samples]
        print(f"Using first {n_samples} examples (out of {len(examples)} total).\n")
    else:
        print(f"Using all {len(examples)} examples for evaluation.\n")

    model_type = "LoRA" if lora else "Full fine-tuned"
    print(f"Running inference on {model_type} model (run: {run_name})...")
    results = generate_batch.remote(run_name, examples, lora)

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
    print(f"EVALUATION RESULTS ({model_type} Model)")
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
        model_suffix = "lora" if lora else "full"
        output_file = f"results/{model_suffix}_{model_name}.json"

    output_data = {
        "run_name": run_name,
        "model_type": model_type,
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

    txt_output_file = output_file.replace(".json", ".txt")
    with open(txt_output_file, "w") as f:
        f.write(f"{'='*80}\n")
        f.write(f"EVALUATION RESULTS ({model_type} Model)\n")
        f.write(f"{'='*80}\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Dataset: {data_file}\n")
        f.write(f"Total examples: {len(results)}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"  Exact Match:       {exact_match:.2%}\n")
        f.write(f"  Has 'Plan:' :      {has_plan_keyword:.2%}\n")
        f.write(f"  Has 'step':        {has_steps:.2%}\n")
        f.write(f"{'='*80}\n\n")

        for i, r in enumerate(results):
            match_status = (
                "✓ CORRECT"
                if r["predicted"].strip() == r["expected"].strip()
                else "✗ INCORRECT"
            )
            f.write(f"\n{'='*80}\n")
            f.write(f"Example {i+1} - {match_status}\n")
            f.write(f"{'='*80}\n\n")

            lines = r["input"].split("\n")
            for line in lines:
                f.write(f"{line}\n")

            f.write(f"\nExpected Plan:\n")
            f.write(f"{r['expected']}\n")

            f.write(f"\nPredicted Plan:\n")
            f.write(f"{r['predicted']}\n")
            f.write(f"\n{'-'*80}\n")

    print(f"Human-readable results saved to {txt_output_file}")

    print(f"\n{'='*60}")
    print("Sample predictions:")
    print(f"{'='*60}\n")
    for i, r in enumerate(results[:3]):
        print(f"Example {i+1}:")
        print(f"Expected:\n{r['expected']}\n")
        print(f"Predicted:\n{r['predicted']}\n")
        print(f"{'-'*60}\n")
