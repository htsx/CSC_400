import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# File paths
skytrax_dataset = "../../data/webscrapper/cleaned_balanced.csv"
predictions_file = "../../data/deep_learning/vanilla/dl_results.csv"

try:
    # Load the ground truth data and drop rows with NaN values
    df_truth = pd.read_csv(skytrax_dataset).dropna(subset=['review_classification'])

    print("Columns in ground truth dataset:", df_truth.columns)
    print("First few rows of ground truth dataset:\n", df_truth.head())

    # Load predictions
    df_pred = pd.read_csv(predictions_file).reset_index(drop=True)

    print("Columns in predicted sentiment dataset:", df_pred.columns)
    print("First few rows of predicted sentiment dataset:\n", df_pred.head())

    # Apply the 12,000 limit consistently
    MAX_REVIEWS = 12000
    df_truth = df_truth.head(MAX_REVIEWS).reset_index(drop=True)
    df_pred = df_pred.head(MAX_REVIEWS).reset_index(drop=True)

    # Check for dataset length mismatch
    if len(df_truth) != len(df_pred):
        raise ValueError(f"Mismatch in number of reviews: {len(df_truth)} in ground truth vs {len(df_pred)} in predictions.")

    # Clean labels
    y_true = df_truth['review_classification'].astype(str).str.strip().str.capitalize()
    y_pred = df_pred['sentiment'].astype(str).str.strip().str.capitalize()

    sentiment_labels = ['Negative', 'Neutral', 'Positive']

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    print("Unique ground truth labels:", y_true.unique())
    print("Unique predicted labels:", y_pred.unique())

    print("\nEvaluation Metrics for the first 12,000 reviews:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save metrics
    metrics_data = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv("../../data/deep_learning/vanilla/dl_evaluation_metrics.csv", index=False)

    # Classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=sentiment_labels, zero_division=0)
    print(report)

    report_data = classification_report(y_true, y_pred, target_names=sentiment_labels, output_dict=True)
    df_report = pd.DataFrame(report_data).transpose()
    df_report.to_csv("../../data/deep_learning/vanilla/dl_classification_report.csv", index=True)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=sentiment_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    df_conf_matrix = pd.DataFrame(conf_matrix, index=sentiment_labels, columns=sentiment_labels)
    df_conf_matrix.to_csv("../../data/deep_learning/vanilla/dl_confusion_matrix.csv", index=True)

    # Misclassified
    df_misclassified = df_truth.copy()
    df_misclassified['predicted_sentiment'] = y_pred
    df_misclassified = df_misclassified[
        df_misclassified['review_classification'].str.strip().str.capitalize() != df_misclassified['predicted_sentiment']
    ]

    if not df_misclassified.empty:
        print(f"\nNumber of misclassified reviews: {len(df_misclassified)}")
        print("\nSample of misclassified reviews (Skytrax Dataset vs Prediction):")
        
        sample = df_misclassified[['review_text', 'review_classification', 'predicted_sentiment']].head(10)
        for index, row in sample.iterrows():
            review_text = row['review_text']
            if len(review_text) > 100:
                review_text = review_text[:100] + '...'
            print(f"Review: {review_text}\nTrue Sentiment: {row['review_classification']} | Predicted Sentiment: {row['predicted_sentiment']}\n")

        df_misclassified.to_csv("../../data/deep_learning/vanilla/dl_misclassified_reviews.csv", index=False)
        print("\nAll misclassified reviews saved to 'dl_misclassified_reviews.csv'")
    else:
        print("No misclassified reviews found.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except ValueError as e:
    print(f"Data error: {e}")
    exit()
