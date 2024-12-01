# %% MODULE BEGINS
module_name = 'module_PA4'

'''
Version: 1.1

Description:
    Model and Assessment with Visual Comparisons.

Authors:
    NeNai: Olisemeka Nmarkwe and Sujana Mehta. (W0762669 and W0757459 respectively)

Date Created     :  11/25/2024
Date Last Updated:  12/1/2024

Doc:


Notes:

'''

# %% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, confusion_matrix, f1_score

# %% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# File names
file_names = [
    "TestData_CPz.csv",
    "TestData_M1.csv",
    "TestData_M2.csv",
    "TrainValidateData_CPz.csv",
    "TrainValidateData_M1.csv",
    "TrainValidateData_M2.csv"
]

# Dictionary to store results
results = {}

# %% FUNCTION DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load the files into a dictionary of DataFrames
data_frames = {file_name: pd.read_csv(file_name) for file_name in file_names}

# Check the loaded DataFrames (optional)
for name, df in data_frames.items():
    print(f"Preview of {name}:")
    print(df.head(), "\n")

# Function to compute clustering metrics
def calculate_metrics(data, targets, n_clusters=2):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[features])
    centroids = kmeans.cluster_centers_
    
    # 1. Cohesion
    cohesion = kmeans.inertia_  
    
    # 2. Separation
    separation = sum(
        sum((centroids[i] - centroids[j]) ** 2) 
        for i in range(n_clusters) for j in range(i+1, n_clusters)
    )
    
    # 3. Combined cohesion and separation (Silhouette Score)
    silhouette = silhouette_score(data[features], data['cluster'])
    
    # 4. Recall, Specificity, and F1-measure
    # Map true targets to numeric values for confusion matrix
    true_labels = (data[targets] == 'Sb2').astype(int)
    pred_labels = data['cluster']
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    
    # Recall, Specificity, F1-Measure
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Return metrics
    return {
        'Cohesion': cohesion,
        'Separation': separation,
        'Silhouette Score': silhouette,
        'Recall': recall,
        'Specificity': specificity,
        'F1-Measure': f1
    }

# Process each file
for file_name in file_names:
    df = pd.read_csv(file_name)
    
    # Features and target column
    features = [col for col in df.columns if col.startswith('f')]
    target_col = 'target'
    
    # Calculate metrics
    metrics = calculate_metrics(df, target_col, n_clusters=2)
    results[file_name] = metrics

# Function to plot clustering results
def plot_metrics(results):
    metric_names = list(results[next(iter(results))].keys())
    
    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=list(results.keys()), y=[results[file][metric] for file in results.keys()])
        plt.title(f'Comparison of {metric}')
        plt.xlabel('Datasets')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add numeric labels to each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        fontsize=10, color='black', 
                        xytext=(0, 10), textcoords='offset points')
        
        # Save the plot 
        plot_file = f"{module_name}_metric_{metric}.png"
        plt.savefig(plot_file)
        print(f"Saved plot for {metric} as {plot_file}")
        
        plt.show()

# Function to visualize clusters 
def visualize_clusters(data_frames):
    for file_name, df in data_frames.items():
        features = [col for col in df.columns if col.startswith('f')]
        target_col = 'target'
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[features])
        
        # Plot clustering results
        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['cluster'], palette='viridis', style=df[target_col])
        plt.title(f'Cluster Visualization for {file_name}')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend(title='Cluster')
        plt.tight_layout()
        
        # Save the plot as a file
        plot_file = f"{module_name}_cluster_{file_name}.png"
        plt.savefig(plot_file)
        print(f"Saved cluster plot for {file_name} as {plot_file}")
        
        plt.show()

# %% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    
    # Display results
    for file, metrics in results.items():
        print(f"Metrics for {file}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    # Plot evaluation metrics
    print("Plotting evaluation metrics...")
    plot_metrics(results)
    
    # Visualize clusters
    print("Visualizing clusters...")
    visualize_clusters(data_frames)

    print("Model and assessment completed.")
# %%
