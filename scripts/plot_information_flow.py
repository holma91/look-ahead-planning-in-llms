"""
Generate Figure 5 replication plots from information flow analysis results.

Usage:
    uv run scripts/plot_information_flow.py
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE_CHUNKS = ["init_token", "init_state", "goal_token", "goal_state", "plan_token"]


def generate_paper_plots(json_file: str, output_path: str):
    """Generate normalized heatmaps matching Figure 5 from the paper."""
    print(f"Loading {json_file}...")
    with open(json_file, "r") as f:
        data = json.load(f)

    results = data["results"]
    num_layers = data["num_layers"]
    steps = sorted(results.keys(), key=lambda x: int(x.split("_")[1]))[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for step_idx, step_key in enumerate(steps):
        ax = axes[step_idx]
        step_num = int(step_key.split("_")[1])

        # Build chunk list: base + history_1..N-1 + action_prompt
        chunks = (
            BASE_CHUNKS
            + [f"history_{i}" for i in range(1, step_num)]
            + ["action_prompt"]
        )

        # Build data matrix [num_chunks x num_layers]
        matrix = np.zeros((len(chunks), num_layers))
        for i, chunk in enumerate(chunks):
            for layer in range(num_layers):
                layer_key = str(layer)
                if (
                    layer_key in results[step_key]
                    and chunk in results[step_key][layer_key]
                ):
                    matrix[i, layer] = results[step_key][layer_key][chunk]

        # Normalize per layer (column-wise)
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        matrix = matrix / col_sums

        # Plot
        im = ax.imshow(matrix, aspect="auto", cmap="Purples", vmin=0, vmax=1.0)
        ax.set_title(f"step {step_num}", fontsize=16)
        ax.set_yticks(range(len(chunks)))
        ax.set_yticklabels([c.replace("_", " ") for c in chunks], fontsize=10)
        ax.set_xticks(range(0, num_layers, 5))
        ax.set_xticklabels(range(0, num_layers, 5))

    # Leave space on the right for the colorbar
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # Add colorbar in the reserved space
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    files = sorted(glob.glob("results/information_flow_full_*.json"))
    if files:
        generate_paper_plots(
            files[-1], "results/information_flow_paper_replication.png"
        )
    else:
        print("No result files found in results/")
