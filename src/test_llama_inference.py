import modal

app = modal.App("llama2-inference-test")

# Create a volume to cache models
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
    volumes={"/cache": model_cache},  # Mount the volume
    timeout=600,
)
def generate(prompt: str):
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # Reload volume to get any cached models
    model_cache.reload()

    # Set cache directory to use the persistent volume
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
    model_cache.commit()  # Save the downloaded model to persistent storage
    print("Model loaded!")

    print(f"Generating response for prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


@app.local_entrypoint()
def main(prompt: str = "The capital of France is"):
    print(f"Testing Llama-2-7b inference on Modal...")
    print(f"Prompt: {prompt}\n")

    result = generate.remote(prompt)

    print(f"\n{'='*60}")
    print(f"Full output:\n{result}")
    print(f"{'='*60}")
