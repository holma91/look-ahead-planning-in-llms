"""
Quick single-example test for trained LoRA models.

Usage:
    # Test with default L1 example
    uv run modal run src.quick_inference_test --run-name axo-2025-11-17-08-26-35-b9fa

    # Test with custom prompt
    uv run modal run src.quick_inference_test --run-name <run-name> --prompt "your prompt"
"""

import modal

app = modal.App("quick-inference-test")

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
    volumes={"/cache": model_cache, "/runs": runs_volume},
    timeout=600,
)
def generate_with_lora(run_name: str, prompt: str):
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

    print(f"\n{'='*60}")
    print(f"PROMPT:\n{prompt}")
    print(f"{'='*60}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt) :].strip()

    print(f"{'='*60}")
    print(f"RESPONSE:\n{response}")
    print(f"{'='*60}\n")

    return response


@app.local_entrypoint()
def main(
    run_name: str = "axo-2025-11-17-08-26-35-b9fa",
    prompt: str = "",
):
    if not prompt:
        prompt = """[INST] Given the rules, initial state, and goal state, generate an optimal plan to transform the initial state into the goal state.

Rule:
You can pick-up color1. You can stack color1 on-top-of color2. You can stack color1 on-top-of table.

Init state:
<blue on yellow>
<red on green>

Goal state:
<red on blue on yellow>
<green> [/INST]"""

    print(f"🧪 Quick test: {run_name}\n")
    response = generate_with_lora.remote(run_name, prompt)
    print(f"✅ Complete!")
