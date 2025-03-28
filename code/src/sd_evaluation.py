import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load ground truth and predicted sentiment files
ground_truth_file = "ground_truth_reviews.csv"
predictions_file = "combined_results.csv"

try:
    # Load ground truth data and drop NaN values
    df_truth = pd.read_csv(ground_truth_file).dropna(subset=['ground_truth_sentiment'])
    
    # Load combined results and ensure it matches ground truth in length
    df_pred = pd.read_csv(predictions_file).head(len(df_truth))  # Take the same number of rows as ground truth
    
    # Print the number of rows after dropping NaN
    print(f"Number of ground truth reviews: {len(df_truth)}")
    print(f"Number of predicted reviews: {len(df_pred)}")

    # Ensure the number of reviews match
    if len(df_truth) != len(df_pred):
        raise ValueError("Mismatch in number of reviews between ground truth and predictions.")

    # Extract ground truth labels and predicted labels
    y_true = df_truth['ground_truth_sentiment'].astype(str).str.strip().str.capitalize()
    y_pred = df_pred['Combined_Sentiment'].astype(str).str.strip().str.capitalize()

    # Print unique values in both to detect mismatches
    print("\nGround truth labels:", y_true.unique())
    print("Predicted labels:", y_pred.unique())

    # Calculate evaluation metrics with zero division handling
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    # Print results
    print("\nEvaluation Metrics for Combined Sentiment Analysis (First 1000 Reviews):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Generate a detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Positive', 'Negative', 'Neutral'], zero_division=0))

    # Generate confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative', 'Neutral']))

    # Find misclassified reviews
    df_misclassified = df_truth.copy()
    df_misclassified['predicted_sentiment'] = y_pred
    df_misclassified = df_misclassified[df_misclassified['ground_truth_sentiment'] != df_misclassified['predicted_sentiment']]

    # Display misclassified reviews with both ground truth and predicted sentiment
    if not df_misclassified.empty:
        print(f"\nNumber of misclassified reviews: {len(df_misclassified)}")
        print("\nSample of misclassified reviews (Ground Truth vs Prediction):")
        print(df_misclassified[['review_text', 'ground_truth_sentiment', 'predicted_sentiment']].head(10))  # Displaying a sample of 10
        
        # Save misclassified reviews to a file for further analysis
        df_misclassified.to_csv("data/dl_misclassified_reviews.csv", index=False)
        print("\nAll misclassified reviews saved to 'data/dl_misclassified_reviews.csv'")
    else:
        print("No misclassified reviews found.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except ValueError as e:
    print(f"Data error: {e}")
    exit()

