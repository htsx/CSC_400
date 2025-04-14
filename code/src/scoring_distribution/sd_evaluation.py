import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load ground truth and predicted sentiment files
skytrax_dataset = "../../data/webscrapper/cleaned_balanced.csv"
predictions_file = "../../data/scoring_distribution/tweaked/sd_results.csv"

try:
    # Load the ground truth data and drop any rows with NaN in 'review_classification'
    df_truth = pd.read_csv(skytrax_dataset).dropna(subset=['review_classification'])
    
    # Load the predictions from the scoring distribution (our model's output)
    df_pred = pd.read_csv(predictions_file).reset_index(drop=True)

    # Make sure the number of reviews match between the ground truth and predictions
    if len(df_truth) != len(df_pred):
        raise ValueError(f"Mismatch in the number of reviews: {len(df_truth)} Skytrax Dataset reviews, but {len(df_pred)} predicted reviews.")

    # Extract the actual and predicted sentiment labels and ensure they’re cleaned up (no extra spaces, capitalized)
    y_true = df_truth['review_classification'].astype(str).str.strip().str.capitalize()
    y_pred = df_pred['review_classification'].astype(str).str.strip().str.capitalize()

    # Define sentiment categories for our analysis
    sentiment_labels = ['Negative', 'Neutral', 'Positive']

    # Calculate the evaluation metrics, making sure we handle any zero division errors
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    # Print out unique labels for both ground truth and predictions (just to see what we’re working with)
    print("Unique ground truth labels:", y_true.unique())
    print("Unique predicted labels:", y_pred.unique())

    # Display the evaluation metrics
    print("\nEvaluation Metrics for the reviews:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save the evaluation metrics to a CSV file for later reference
    metrics_data = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv("../../data/scoring_distribution/tweaked/sd_evaluation_metrics.csv", index=False)

    # Generate a detailed classification report to see how the model did for each class
    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=sentiment_labels, zero_division=0)
    print(report)

    # Save the classification report as a CSV for documentation
    report_data = classification_report(y_true, y_pred, target_names=sentiment_labels, output_dict=True)
    df_report = pd.DataFrame(report_data).transpose()
    df_report.to_csv("../../data/scoring_distribution/tweaked/sd_classification_report.csv", index=True)

    # Generate the confusion matrix to see where the model made mistakes
    conf_matrix = confusion_matrix(y_true, y_pred, labels=sentiment_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Save the confusion matrix as a CSV file
    df_conf_matrix = pd.DataFrame(conf_matrix, index=sentiment_labels, columns=sentiment_labels)
    df_conf_matrix.to_csv("../../data/scoring_distribution/tweaked/sd_confusion_matrix.csv", index=True)

    # Check for misclassified reviews (where the prediction doesn’t match the truth)
    df_misclassified = df_truth.copy()
    df_misclassified['predicted_classification'] = y_pred  # Fix: Adding a new column for predicted labels
    df_misclassified = df_misclassified[df_misclassified['review_classification'].str.strip().str.capitalize() != df_misclassified['predicted_classification']]

    # If we have misclassified reviews, let’s display a sample and save the details
    if not df_misclassified.empty:
        print(f"\nNumber of misclassified reviews: {len(df_misclassified)}")
        print("\nSample of misclassified reviews (Skytrax Dataset vs Prediction):")
        
        # Print a few samples of misclassified reviews (just the first 10)
        sample_misclassified = df_misclassified[['review_text', 'review_classification', 'predicted_classification']].head(10)
        for index, row in sample_misclassified.iterrows():
            review_text = row['review_text']
            if len(review_text) > 100:  # If the review is too long, we’ll truncate it
                review_text = review_text[:100] + '...'  # Truncate the text for better readability
            print(f"Review: {review_text}\nTrue Sentiment: {row['review_classification']} | Predicted Sentiment: {row['predicted_classification']}\n")
        
        # Save all the misclassified reviews to a file for further investigation
        df_misclassified.to_csv("../../data/scoring_distribution/tweaked/sd_misclassified_reviews.csv", index=False)
        print("\nAll misclassified reviews saved to 'sd_misclassified_reviews.csv'")

    else:
        print("No misclassified reviews found.")  # If no misclassifications, great!

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except ValueError as e:
    print(f"Data error: {e}")
    exit()
