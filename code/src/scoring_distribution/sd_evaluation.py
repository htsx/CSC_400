import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load ground truth and predicted sentiment files
ground_truth_file = "../../data/ground_truth/ground_truth_reviews.csv"
predictions_file = "../../data/scoring_distribution/combined_results.csv"

try:
    # Load ground truth data and drop NaN values
    df_truth = pd.read_csv(ground_truth_file).dropna(subset=['ground_truth_sentiment'])
    
    # Limit the data to the first 1001 reviews
    df_truth = df_truth.head(1001)
    
    # Load combined results and ensure it matches ground truth in length
    df_pred = pd.read_csv(predictions_file).head(len(df_truth))  # Take the same number of rows as ground truth
    
    # Ensure the number of reviews match
    if len(df_truth) != len(df_pred):
        raise ValueError("Mismatch in number of reviews between ground truth and predictions.")

    # Extract ground truth labels and predicted labels, ensure they are stripped of any spaces and capitalized
    y_true = df_truth['ground_truth_sentiment'].astype(str).str.strip().str.capitalize()
    y_pred = df_pred['Combined_Sentiment'].astype(str).str.strip().str.capitalize()

    # Calculate evaluation metrics with zero division handling
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    # Print results
    print("\nEvaluation Metrics for the first 1001 reviews:")
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
    df_metrics.to_csv("../../data/scoring_distribution/evaluation_results/sd_evaluation_metrics.csv", index=False)

    # Generate a detailed classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=['Positive', 'Negative', 'Neutral'], zero_division=0)
    print(report)

    # Save classification report as CSV
    report_data = classification_report(y_true, y_pred, target_names=['Positive', 'Negative', 'Neutral'], output_dict=True)
    df_report = pd.DataFrame(report_data).transpose()
    df_report.to_csv("../../data/scoring_distribution/evaluation_results/sd_classification_report.csv", index=True)

    # Generate confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative', 'Neutral'])
    print(conf_matrix)

    # Save confusion matrix to a CSV file
    df_conf_matrix = pd.DataFrame(conf_matrix, index=['Positive', 'Negative', 'Neutral'], columns=['Positive', 'Negative', 'Neutral'])
    df_conf_matrix.to_csv("../../data/scoring_distribution/evaluation_results/sd_confusion_matrix.csv", index=True)

    # Find misclassified reviews
    df_misclassified = df_truth.copy()
    df_misclassified['predicted_sentiment'] = y_pred
    df_misclassified = df_misclassified[df_misclassified['ground_truth_sentiment'].str.strip().str.capitalize() != df_misclassified['predicted_sentiment']]

    # Display misclassified reviews with both ground truth and predicted sentiment
    if not df_misclassified.empty:
        print(f"\nNumber of misclassified reviews: {len(df_misclassified)}")
        print("\nSample of misclassified reviews (Ground Truth vs Prediction):")
        print(df_misclassified[['review_text', 'ground_truth_sentiment', 'predicted_sentiment']].head(10))  # Displaying a sample of 10
        
        # Save misclassified reviews to a file for further analysis
        df_misclassified.to_csv("../../data/scoring_distribution/evaluation_results/sd_misclassified_reviews.csv", index=False)
        print("\nAll misclassified reviews saved to 'sd_misclassified_reviews.csv'")
    else:
        print("No misclassified reviews found.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except ValueError as e:
    print(f"Data error: {e}")
    exit()
