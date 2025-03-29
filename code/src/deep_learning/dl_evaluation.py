import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load ground truth and predicted sentiment files
ground_truth_file = "data/ground_truth/ground_truth_reviews.csv"
predictions_file = "data/deep_learning/dl_results.csv"

# Load datasets
df_truth = pd.read_csv(ground_truth_file).head(1000)  # Use only the first 1000 labeled reviews
df_pred = pd.read_csv(predictions_file).head(1000)    # Ensure the same 1000 reviews are used

# Ensure the number of reviews match after loading
if len(df_truth) != len(df_pred):
    # Align indices by keeping the same set of reviews that are in both datasets
    common_indices = df_truth.index.intersection(df_pred.index)
    df_truth = df_truth.loc[common_indices]
    df_pred = df_pred.loc[common_indices]

# Normalize labels: convert to uppercase and strip whitespace
df_truth['ground_truth_sentiment'] = df_truth['ground_truth_sentiment'].astype(str).str.upper().str.strip()
df_pred['sentiment'] = df_pred['sentiment'].astype(str).str.upper().str.strip()

# Extract ground truth labels and predicted labels
y_true = df_truth['ground_truth_sentiment']
y_pred = df_pred['sentiment']

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Print results
print("Evaluation Metrics for Deep Learning-Based Sentiment Analysis (First 1000 Reviews):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save the evaluation metrics to a CSV file
metrics_data = {
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1]
}
df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_csv("data/deep_learning/evaluation_results/dl_evaluation_metrics.csv", index=False)

# Generate a detailed classification report
print("\nDetailed Classification Report:")
report = classification_report(y_true, y_pred, target_names=['POSITIVE', 'NEGATIVE', 'NEUTRAL'], zero_division=0)
print(report)

# Save classification report as CSV
report_data = classification_report(y_true, y_pred, target_names=['POSITIVE', 'NEGATIVE', 'NEUTRAL'], output_dict=True)
df_report = pd.DataFrame(report_data).transpose()
df_report.to_csv("data/deep_learning/evaluation_results/dl_classification_report.csv", index=True)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
print("\nConfusion Matrix:")
print(conf_matrix)

# Save confusion matrix to a CSV file
df_conf_matrix = pd.DataFrame(conf_matrix, index=['POSITIVE', 'NEGATIVE', 'NEUTRAL'], columns=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
df_conf_matrix.to_csv("data/deep_learning/evaluation_results/dl_confusion_matrix.csv", index=True)

# Find misclassified reviews
df_misclassified = df_truth.copy()
df_misclassified['predicted_sentiment'] = df_pred['sentiment']
df_misclassified = df_misclassified[df_misclassified['ground_truth_sentiment'] != df_misclassified['predicted_sentiment']]

# Display misclassified reviews with both ground truth and predicted sentiment
if not df_misclassified.empty:
    print(f"\nNumber of misclassified reviews: {len(df_misclassified)}")
    print("\nSample of misclassified reviews (Ground Truth vs Prediction):")
    print(df_misclassified[['review_text', 'ground_truth_sentiment', 'predicted_sentiment']])
    
    # Save misclassified reviews to a file for further analysis
    df_misclassified.to_csv("data/deep_learning/evaluation_results/dl_misclassified_reviews.csv", index=False)
    print("\nAll misclassified reviews saved to 'dl_misclassified_reviews.csv'")
else:
    print("No misclassified reviews found.")
