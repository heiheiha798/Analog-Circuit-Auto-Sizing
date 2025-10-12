import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import re

def parse_data_from_file(file_path):
    """
    Parses the evaluation results from a text file.
    """
    data = []
    current_iter_data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Start of a new iteration block
            if line.startswith("========== Evaluation result of iter"):
                if current_iter_data:
                    data.append(current_iter_data)
                current_iter_data = {}
                match = re.search(r'iter (\d+)', line)
                if match:
                    current_iter_data['iter'] = match.group(1)
            
            # Key-value pairs
            elif line.startswith("====== "):
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].replace("====== ", "").strip()
                    value_str = parts[1].strip()
                    
                    # Special handling for "inf" and "-inf"
                    if value_str.startswith("-inf"):
                        value = -np.inf
                    elif value_str.startswith("inf"):
                        value = np.inf
                    else:
                        # Extract the numeric part of the string
                        numeric_match = re.search(r"(-?\d+\.?\d*e?[-+]?\d*)", value_str)
                        if numeric_match:
                            value = float(numeric_match.group(1))
                        else:
                            value = np.nan

                    # Unit conversion for I_OPA
                    if key == 'I_OPA' and value is not np.nan:
                        value *= 1000

                    current_iter_data[key] = value

            # Fitness value
            elif line.startswith("==== iter"):
                match = re.search(r'fitness: (.+)', line)
                if match:
                    fitness_str = match.group(1).split()[0]
                    if fitness_str.lower() == 'inf':
                        current_iter_data['Fitness'] = np.inf
                    else:
                        current_iter_data['Fitness'] = float(fitness_str)

    # Append the last block of data
    if current_iter_data:
        data.append(current_iter_data)

    return data

def plot_results(data, output_dir="plots"):
    """
    Plots the performance metrics and fitness as line graphs based on provided data.
    """
    if not data:
        print("No data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    fitness_values = [item.get('Fitness') for item in data]
    
    metrics = {
        'UGB': 'Hz',
        'Phase_Margin': 'deg',
        'Gain_db': 'db',
        'Gain_Margin': 'deg',
        'I_OPA': 'mA',
        'Total_Area': 'um^2'
    }
    
    # Replace inf with a very negative or positive value for plotting
    plot_data = {}
    for key in metrics:
        plot_data[key] = []
        for item in data:
            val = item.get(key)
            if isinstance(val, (np.ndarray, list)):
                 val = val[0]
            if val == -np.inf:
                val = -1e9
            elif val == np.inf:
                val = 1e9
            
            plot_data[key].append(val)
                
    # Create plots
    for key, unit in metrics.items():
        if not plot_data[key] or all(v == plot_data[key][0] for v in plot_data[key]):
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(fitness_values, plot_data[key])
        
        ax.set_title(f"Evolution of {key} vs. Fitness")
        ax.set_xlabel("Fitness (log scale)")
        ax.set_ylabel(f"{key} ({unit})")
        ax.grid(True)
        
        # Use log scale for fitness
        ax.set_xscale('log')

        # Adjust y-axis for UGB and Phase_Margin to show -inf
        if key in ['UGB', 'Phase_Margin']:
            min_val = min(v for v in plot_data[key] if v > -1e9) if any(v > -1e9 for v in plot_data[key]) else -1e9
            max_val = max(plot_data[key])
            ax.set_ylim(bottom=min(min_val, -10), top=max(max_val, 10))
            
            # Use log scale for UGB if values are very large
            if key == 'UGB' and max(plot_data[key]) > 1000:
                 ax.set_yscale('symlog', linthresh=1000)

        # Adjust y-axis for Gain_Margin
        if key == 'Gain_Margin':
            min_val = min(v for v in plot_data[key] if v < 1e9) if any(v < 1e9 for v in plot_data[key]) else 1e9
            max_val = max(plot_data[key])
            ax.set_ylim(bottom=min(min_val, -150), top=max(max_val, 10))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}_vs_fitness_plot.png"))
        plt.close(fig)

if __name__ == '__main__':
    file_path = "1.txt"
    if os.path.exists(file_path):
        data = parse_data_from_file(file_path)
        # Filter out data points with infinite fitness for plotting on a log scale
        finite_data = [d for d in data if d.get('Fitness') not in [np.inf, -np.inf]]
        plot_results(finite_data)
        print(f"Plots have been generated in the 'plots' directory based on data from {file_path}.")
    else:
        print(f"Error: The file '{file_path}' was not found in the current directory.")
