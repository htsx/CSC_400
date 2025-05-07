import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the paths for each method's results
paths = {
    'Deep Learning': '../data/deeplearning/',
    'Rule-Based': '../data/rulebased/',
    'Word Scoring': '../data/wordscoring/'
}

# Initialize lists to store the results
methods = []
accuracy = []
precision = []
recall = []
f1_score = []

# Function to extract metrics from evaluation CSV files
def extract_metrics(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract the accuracy, precision, recall, and F1-score from the relevant columns
        metrics = {
            'accuracy': df['Accuracy'].values[0],  # Extract accuracy value
            'precision': df['Precision'].values[0],  # Extract precision value
            'recall': df['Recall'].values[0],  # Extract recall value
            'f1_score': df['F1 Score'].values[0]  # Extract f1 score value
        }
        return metrics
    except Exception as e:
        print(f"Error extracting metrics from {file_path}: {e}")
        return None

# Iterate over the methods and extract metrics from the relevant files
for method, path in paths.items():
    if method == 'Deep Learning':
        # Construct the file path for deep learning evaluation metrics CSV
        evaluation_file = os.path.join(path, "t-dl_evaluation_metrics.csv")
    elif method == 'Rule-Based':
        # Construct the file path for rule-based evaluation metrics CSV
        evaluation_file = os.path.join(path, "t-r_evaluation_metrics.csv")
    elif method == 'Word Scoring':
        # Construct the file path for word scoring evaluation metrics CSV
        evaluation_file = os.path.join(path, "t-wd_evaluation_metrics.csv")
    
    # Extract metrics from the evaluation CSV
    metrics = extract_metrics(evaluation_file)
    
    if metrics:
        # Append to the results lists
        methods.append(method)
        accuracy.append(metrics.get('accuracy', 0))
        precision.append(metrics.get('precision', 0))
        recall.append(metrics.get('recall', 0))
        f1_score.append(metrics.get('f1_score', 0))

# Create a DataFrame to store all results for easy plotting
results_df = pd.DataFrame({
    'Method': methods,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score
})

# Set the plot style
sns.set(style="whitegrid")

# Define a color palette for the different methods
color_palette = sns.color_palette("Set2", len(paths))

# Plot for Accuracy
plt.figure(figsize=(8, 6))
sns.barplot(x='Method', y='Accuracy', data=results_df, palette=color_palette)
plt.title('Accuracy Comparison')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('accuracy_comparison.png')  # Save as image
plt.show()

# Plot for Precision
plt.figure(figsize=(8, 6))
sns.barplot(x='Method', y='Precision', data=results_df, palette=color_palette)
plt.title('Precision Comparison')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('precision_comparison.png')  # Save as image
plt.show()

# Plot for Recall
plt.figure(figsize=(8, 6))
sns.barplot(x='Method', y='Recall', data=results_df, palette=color_palette)
plt.title('Recall Comparison')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('recall_comparison.png')  # Save as image
plt.show()

# Plot for F1-Score
plt.figure(figsize=(8, 6))
sns.barplot(x='Method', y='F1-Score', data=results_df, palette=color_palette)
plt.title('F1-Score Comparison')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('f1_score_comparison.png')  # Save as image
plt.show()
