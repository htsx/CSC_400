import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load ground truth and predicted sentiment files
ground_truth_file = "data/ground_truth_reviews.csv"
predictions_file = "data/dl_results.csv"

# Load datasets
df_truth = pd.read_csv(ground_truth_file).head(1000)  # Use only the first 1000 labeled reviews
df_pred = pd.read_csv(predictions_file).head(1000)    # Ensure the same 1000 reviews are used

# Ensure the number of reviews match
if len(df_truth) != len(df_pred):
    raise ValueError("Mismatch in number of reviews between ground truth and predictions.")

# Extract ground truth labels and predicted labels
y_true = df_truth['ground_truth_sentiment']
y_pred = df_pred['sentiment']

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')  # Macro-average for balanced class importance
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Print results
print("Evaluation Metrics for Deep Learning-Based Sentiment Analysis (First 1000 Reviews):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
