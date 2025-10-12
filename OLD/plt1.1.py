import os
import re
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

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

def find_minority_clusters_per_metric(data, n_clusters=3, minority_threshold=10):
    """
    Finds and prints minority data points for each metric relative to fitness.
    """
    if not data:
        print("No data to process.")
        return

    # Filter out data with infinite fitness for meaningful clustering
    finite_data = [d for d in data if d.get('Fitness') not in [np.inf, -np.inf]]
    if not finite_data:
        print("No data with finite fitness to cluster.")
        return

    # Key order for consistent feature vectors
    metrics = ['UGB', 'Phase_Margin', 'Gain_db', 'Gain_Margin', 'I_OPA', 'Total_Area']
    
    for metric in metrics:
        print("-" * 50)
        print(f"Analyzing {metric} vs. Fitness")
        print("-" * 50)

        # Create feature set for clustering
        features = []
        for d in finite_data:
            metric_val = d.get(metric)
            fitness_val = d.get('Fitness')
            features.append([metric_val, fitness_val])
        
        X = np.array(features)

        # Handle infinite values by replacing them with a large number for clustering
        X[X == np.inf] = 1e12
        X[X == -np.inf] = -1e12

        # Check for meaningful data variance
        if np.isclose(np.std(X[:, 0]), 0) or np.isclose(np.std(X[:, 1]), 0):
            unique_count = len(np.unique(X, axis=0))
            if unique_count <= 1:
                print(f"Skipping analysis: {metric} values are too similar for clustering.")
                continue

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_
        
        # Count the number of points in each cluster
        cluster_counts = Counter(labels)
        
        # Find minority clusters
        minority_clusters = {cluster for cluster, count in cluster_counts.items() if count < minority_threshold}

        print("Cluster sizes:")
        for cluster, count in sorted(cluster_counts.items()):
            is_minority = " (Minority)" if cluster in minority_clusters else ""
            print(f"  Cluster {cluster}: {count} points{is_minority}")
        
        if not minority_clusters:
            print(f"No minority clusters found for {metric} based on the threshold.")
            continue
            
        print("\nMinority data points found:")
        
        for i, d in enumerate(finite_data):
            if labels[i] in minority_clusters:
                print(f"========== Iteration {d['iter']} ==========")
                print(f"====== {metric}     : {d[metric]:.4f}")
                print(f"==== Fitness: {d['Fitness']:.4f}")
                print("-" * 20)
    print("---------------------------------------------------------------------")


if __name__ == '__main__':
    file_path = "1.txt"
    if os.path.exists(file_path):
        data = parse_data_from_file(file_path)
        find_minority_clusters_per_metric(data)
    else:
        print(f"Error: The file '{file_path}' was not found.")
