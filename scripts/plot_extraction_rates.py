"""
Generate extraction rates plot from analysis results.

Usage:
    uv run scripts/plot_extraction_rates.py
"""

import json
import glob
import matplotlib.pyplot as plt


def generate_extraction_plot(json_file: str, output_path: str):
    """Generate extraction rates plot matching Figure 3 from the paper."""
    print(f"Loading {json_file}...")
    with open(json_file, "r") as f:
        data = json.load(f)

    mlp_rates = data["results"]["mlp"]
    mhsa_rates = data["results"]["mhsa"]
    layer_rates = data["results"]["layer"]
    num_layers = data["num_layers"]

    print(f"Loaded data for {num_layers} layers")
    print(f"Number of examples: {data['num_examples']}")
    print(f"Stats: {data['stats']}")

    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(num_layers))

    # MLP: blue dashed line
    ax.plot(layers, mlp_rates, "--", linewidth=2.5, label="MLP", color="#1f77b4")
    # MHSA: orange solid line
    ax.plot(layers, mhsa_rates, "-", linewidth=2.5, label="MHSA", color="#ff7f0e")
    # Layer: green dotted line
    ax.plot(layers, layer_rates, ":", linewidth=3, label="Layer", color="#2ca02c")

    ax.set_xlabel("layers", fontsize=18)
    ax.set_ylabel("extraction rates", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=16, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-1, num_layers)

    plt.title(
        "Extraction rate of different components in Llama-2-7b-chat-hf (Full Fine-tuned)",
        fontsize=14,
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    # Print key observations
    print("\nKey observations:")
    print(f"- Layer output extraction rate at final layer: {layer_rates[-1]:.2%}")
    print(
        f"- Peak MHSA extraction rate: {max(mhsa_rates):.2%} at layer {mhsa_rates.index(max(mhsa_rates))}"
    )
    print(
        f"- Peak MLP extraction rate: {max(mlp_rates):.2%} at layer {mlp_rates.index(max(mlp_rates))}"
    )
    print(
        f"\n- MHSA starts extracting at layer: {next((i for i, v in enumerate(mhsa_rates) if v > 0.01), None)}"
    )
    print(
        f"- MLP starts extracting at layer: {next((i for i, v in enumerate(mlp_rates) if v > 0.01), None)}"
    )
    print(
        f"- Layer output reaches >90% at layer: {next((i for i, v in enumerate(layer_rates) if v > 0.9), None)}"
    )


if __name__ == "__main__":
    # Find the latest extraction rates JSON file
    json_files = glob.glob("results/extraction_rates_*.json")
    if not json_files:
        print("No extraction rates JSON files found in results/")
        exit(1)

    latest_file = max(json_files)
    generate_extraction_plot(latest_file, "results/extraction_rates_plot.png")
