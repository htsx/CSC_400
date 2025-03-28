import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load ground truth and predicted sentiment files
ground_truth_file = "../../data/ground_truth/ground_truth_reviews.csv"
predictions_file = "../../data/keywords_topics/kt_results.csv"

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
print("Evaluation Metrics for Keyword and Topic-Based Sentiment Analysis (First 1000 Reviews):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Generate a detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['POSITIVE', 'NEGATIVE', 'NEUTRAL'], zero_division=0))

# Generate confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL']))

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
    df_misclassified.to_csv("../../data/keywords_topics/evaluation_results/kt_misclassified_reviews.csv", index=False)
    print("\nAll misclassified reviews saved to 'kt_misclassified_reviews.csv'")
else:
    print("No misclassified reviews found.")

# Additional placeholders for keyword/topic-based analysis (insert your logic here)
# Example:
# keywords = ['example_keyword1', 'example_keyword2']  # Replace with actual keywords
# topics = ['example_topic1', 'example_topic2']  # Replace with actual topics

# (Use the above variables in your analysis as needed)

