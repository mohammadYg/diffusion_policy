import json
import matplotlib.pyplot as plt
import os

def load_data(filename):
    """Loads JSON data from a specified file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}. Please ensure the JSON files are in the same directory.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filename}.")
        return None

def plot_performance(data, num_demos):
    """Generates and saves a plot of noise prediction error vs. epochs."""
    if not data or not data.get("epochs"):
        print(f"Skipping plot for num_demos={num_demos} due to missing or incomplete data.")
        return

    epochs = data["epochs"]
    
    # --- Start Plotting ---
    plt.figure(figsize=(12, 7))
    
    # 1. Plot the DM Error (Baseline)
    dm_errors = data.get("DM_noise_pred_error", [])
    if dm_errors:
        plt.plot(epochs, dm_errors, label="DM (Baseline)", linewidth=3, linestyle='--', color='black')
        
    # 2. Plot the PDM Errors for each KL weight
    pdm_data = data.get("PDM_noise_pred_error", [])
    colors = plt.cm.turbo
    num_kl_weights = len(pdm_data)
    
    for i, entry in enumerate(pdm_data):
        kl_weight = entry["kl_weight"]
        errors = entry["noise_pred_errors"]
        
        # We use a color map to distinguish the lines
        color = colors(i / (num_kl_weights - 1) if num_kl_weights > 1 else 0.5)
        
        if errors and len(errors) == len(epochs):
            label = f"PDM (KL={kl_weight})"
            plt.plot(epochs, errors, label=label, linewidth=1.5, color=color)

    # --- Plot Customization ---
    plt.title(f"Noise Prediction Error vs. Epochs (num_demos = {num_demos})", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Noise Prediction Error", fontsize=14)
    
    # Set X-axis ticks to match epochs exactly
    plt.xticks(epochs, rotation=45, ha='right')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="Model / KL Weight", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for side legend
    
    # Save the plot
    output_filename = f"data/outputs/eval/plot_{num_demos}_demos.png"
    plt.savefig(output_filename)
    print(f"Plot saved successfully as '{output_filename}'")
    
if __name__ == '__main__':
    demo_counts = [20, 30, 40, 80, 90, 100]
    
    for num_demos in demo_counts:
        file_name = f"data/outputs/eval/demos={num_demos}.json"
        data = load_data(file_name)
        if data:
            plot_performance(data, num_demos)

    print("\nPlotting process complete.")