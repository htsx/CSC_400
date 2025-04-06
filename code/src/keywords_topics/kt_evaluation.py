import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load ground truth and predicted sentiment files
skytrax_dataset = "../../data/webscrapper/ground_truth_reviews.csv"
predictions_file = "../../data/keywords_topics/keywords_topics_results.csv"  # Placeholder file path for keywords and topics results

try:
    # Load ground truth data and drop NaN values
    df_truth = pd.read_csv(skytrax_dataset).dropna(subset=['review_classification'])
    
    # Limit to 10,800 reviews
    df_truth = df_truth.head(10800).reset_index(drop=True)
    
    # Load predicted results for keywords and topics (this is a placeholder for now)
    df_pred = pd.read_csv(predictions_file).reset_index(drop=True)

    # Limit to 10,800 reviews
    df_pred = df_pred.head(10800).reset_index(drop=True)

    # Ensure the number of reviews match
    if len(df_truth) != len(df_pred):
        raise ValueError(f"Mismatch in number of reviews: {len(df_truth)} Skytrax Dataset reviews, but {len(df_pred)} predicted reviews.")

    # Extract ground truth labels and predicted labels from placeholder column
    y_true = df_truth['review_classification'].astype(str).str.strip().str.capitalize()
    y_pred = df_pred['Combined_Sentiment'].astype(str).str.strip().str.capitalize()  # Placeholder for actual column name

    # Define sentiment categories
    sentiment_labels = ['Negative', 'Neutral', 'Positive']

    # Calculate evaluation metrics with zero division handling
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    print("Unique ground truth labels:", y_true.unique())
    print("Unique predicted labels:", y_pred.unique())

    # Print results
    print("\nEvaluation Metrics for the first 10,800 reviews:")
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
    df_metrics.to_csv("../../data/keywords_topics/evaluation_results/kt_evaluation_metrics.csv", index=False)

    # Generate a detailed classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=sentiment_labels, zero_division=0)
    print(report)

    # Save classification report as CSV
    report_data = classification_report(y_true, y_pred, target_names=sentiment_labels, output_dict=True)
    df_report = pd.DataFrame(report_data).transpose()
    df_report.to_csv("../../data/keywords_topics/evaluation_results/kt_classification_report.csv", index=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=sentiment_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Save confusion matrix to a CSV file
    df_conf_matrix = pd.DataFrame(conf_matrix, index=sentiment_labels, columns=sentiment_labels)
    df_conf_matrix.to_csv("../../data/keywords_topics/evaluation_results/kt_confusion_matrix.csv", index=True)

    # Find misclassified reviews
    df_misclassified = df_truth.copy()
    df_misclassified['predicted_sentiment'] = y_pred
    df_misclassified = df_misclassified[df_misclassified['review_classification'].str.strip().str.capitalize() != df_misclassified['predicted_sentiment']]

    # Display misclassified reviews with both ground truth and predicted sentiment
    if not df_misclassified.empty:
        print(f"\nNumber of misclassified reviews: {len(df_misclassified)}")
        print("\nSample of misclassified reviews (Skytrax Dataset vs Prediction):")
        
        # Print a sample of misclassified reviews with truncated review text for readability
        sample_misclassified = df_misclassified[['review_text', 'review_classification', 'predicted_sentiment']].head(10)
        for index, row in sample_misclassified.iterrows():
            review_text = row['review_text']
            if len(review_text) > 100:  # Truncate if the review is too long
                review_text = review_text[:100] + '...'  # Truncated text
            print(f"Review: {review_text}\nTrue Sentiment: {row['review_classification']} | Predicted Sentiment: {row['predicted_sentiment']}\n")
        
        # Save misclassified reviews to a file for further analysis
        df_misclassified.to_csv("../../data/keywords_topics/evaluation_results/kt_misclassified_reviews.csv", index=False)
        print("\nAll misclassified reviews saved to 'kt_misclassified_reviews.csv'")

    else:
        print("No misclassified reviews found.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except ValueError as e:
    print(f"Data error: {e}")
    exit()
