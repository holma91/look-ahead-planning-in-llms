import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def generate_paper_plots(json_file, output_dir):
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    results = data['results']
    num_layers = data['num_layers']
    steps = sorted(results.keys(), key=lambda x: int(x.split('_')[1]))
    
    # Setup plot (2 rows x 3 columns for Steps 1-6)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    
    # Define the chunk order EXACTLY as in the paper's Figure 5
    # Note: 'action_prompt' is always the last row
    base_chunks = ['init_token', 'init_state', 'goal_token', 'goal_state', 'plan_token']
    
    print("\nGenerating Figure 5 replication plots...")
    
    for step_idx, step_key in enumerate(steps):
        if step_idx >= 6: break
        
        ax = axes[step_idx]
        step_num = int(step_key.split('_')[1])
        
        # Construct the list of chunks for this specific step
        # 1. Base chunks (init/goal/plan)
        # 2. History chunks (history_1 ... history_{N-1})
        # 3. Action Prompt (current step N)
        
        step_chunks = list(base_chunks)
        
        # Add history for previous completed steps
        # E.g., for Step 3, we add history_1 and history_2
        for i in range(1, step_num):
            step_chunks.append(f'history_{i}')
            
        # Add the action prompt (the current step's prefix)
        step_chunks.append('action_prompt')
        
        # Build data matrix [num_chunks x num_layers]
        raw_matrix = np.zeros((len(step_chunks), num_layers))
        
        for i, chunk in enumerate(step_chunks):
            for layer in range(num_layers):
                layer_key = str(layer)
                # Handle missing keys safely (default to 0.0)
                val = 0.0
                if layer_key in results[step_key] and chunk in results[step_key][layer_key]:
                    val = results[step_key][layer_key][chunk]
                raw_matrix[i, layer] = val
        
        # NORMALIZE column-wise (per layer)
        # Calculates P(attention | layer)
        column_sums = raw_matrix.sum(axis=0)
        column_sums[column_sums == 0] = 1.0  # Prevent divide by zero
        norm_matrix = raw_matrix / column_sums[np.newaxis, :]
        
        # Plot Heatmap
        # cmap='Purples' matches the paper's aesthetic
        im = ax.imshow(norm_matrix, aspect='auto', cmap='Purples', vmin=0, vmax=1.0, interpolation='nearest')
        
        # Styling
        ax.set_title(f'step {step_num}', fontsize=16)
        
        # Y-axis labels
        # Format 'history_1' -> 'history 1' to match paper style exactly
        clean_labels = [l.replace('_', ' ') for l in step_chunks]
        ax.set_yticks(range(len(step_chunks)))
        ax.set_yticklabels(clean_labels, fontsize=10)
        
        # X-axis labels
        ax.set_xlabel('') # Remove label to reduce clutter, or add 'Layer' if preferred
        ax.set_xticks(range(0, num_layers, 5))
        ax.set_xticklabels(range(0, num_layers, 5))
        
        # Tick marks inside/outside
        ax.tick_params(axis='both', which='both', length=4)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    output_path = os.path.join(output_dir, 'information_flow_paper_replication.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    # Find the latest result file automatically
    files = sorted(glob.glob('results/information_flow_full_*.json'))
    if files:
        latest_file = files[-1]
        generate_paper_plots(latest_file, 'results')
    else:
        print("No result files found in results/")

