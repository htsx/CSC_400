import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

#Load the ground truth and predicted sentiment files
skytrax_dataset = "../../data/webscrapper/ground_truth_reviews.csv"
predictions_file = "../../data/deep_learning/dl_results.csv"

try:
    #Load the ground truth data and drop rows with NaN values in 'review_classification'
    df_truth = pd.read_csv(skytrax_dataset).dropna(subset=['review_classification'])
    
    #Print the columns and the first few rows just to make sure it looks right
    print("Columns in ground truth dataset:", df_truth.columns)
    print("First few rows of ground truth dataset:\n", df_truth.head())
    
    #Limit the data to 20000 reviews
    df_truth = df_truth.head(20000).reset_index(drop=True)
    
    #Load the deep learning results
    df_pred = pd.read_csv(predictions_file).reset_index(drop=True)
    
    #Print the columns and first few rows to verify
    print("Columns in predicted sentiment dataset:", df_pred.columns)
    print("First few rows of predicted sentiment dataset:\n", df_pred.head())

    #Limit to 20000 reviews to match the ground truth data
    df_pred = df_pred.head(20000).reset_index(drop=True)

    #Check if both datasets have the same number of reviews
    if len(df_truth) != len(df_pred):
        raise ValueError(f"Mismatch in number of reviews: {len(df_truth)} Skytrax Dataset reviews, but {len(df_pred)} predicted reviews.")

    #Get the ground truth and predicted labels, making sure they're clean (no spaces, capitalized)
    y_true = df_truth['review_classification'].astype(str).str.strip().str.capitalize()
    y_pred = df_pred['sentiment'].astype(str).str.strip().str.capitalize()

    #Define the sentiment labels
    sentiment_labels = ['Negative', 'Neutral', 'Positive']

    #Calculate the evaluation metrics (accuracy, precision, recall, f1) while handling any division by zero
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    print("Unique ground truth labels:", y_true.unique())
    print("Unique predicted labels:", y_pred.unique())

    #Print the evaluation results
    print("\nEvaluation Metrics for the first 10,800 reviews:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    #Save the metrics to a CSV file
    metrics_data = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv("../../data/deep_learning/evaluation_results/dl_evaluation_metrics.csv", index=False)

    #Generate and print a detailed classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=sentiment_labels, zero_division=0)
    print(report)

    #Save the classification report as a CSV
    report_data = classification_report(y_true, y_pred, target_names=sentiment_labels, output_dict=True)
    df_report = pd.DataFrame(report_data).transpose()
    df_report.to_csv("../../data/deep_learning/evaluation_results/dl_classification_report.csv", index=True)

    #Generate the confusion matrix and print it
    conf_matrix = confusion_matrix(y_true, y_pred, labels=sentiment_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    #Save the confusion matrix to a CSV file
    df_conf_matrix = pd.DataFrame(conf_matrix, index=sentiment_labels, columns=sentiment_labels)
    df_conf_matrix.to_csv("../../data/deep_learning/evaluation_results/dl_confusion_matrix.csv", index=True)

    #Find any misclassified reviews by comparing ground truth and predictions
    df_misclassified = df_truth.copy()
    df_misclassified['predicted_sentiment'] = y_pred
    df_misclassified = df_misclassified[df_misclassified['review_classification'].str.strip().str.capitalize() != df_misclassified['predicted_sentiment']]

    #If there are misclassified reviews, display a sample
    if not df_misclassified.empty:
        print(f"\nNumber of misclassified reviews: {len(df_misclassified)}")
        print("\nSample of misclassified reviews (Skytrax Dataset vs Prediction):")
        
        #Print a sample of misclassified reviews with truncated text for easier reading
        sample_misclassified = df_misclassified[['review_text', 'review_classification', 'predicted_sentiment']].head(10)
        for index, row in sample_misclassified.iterrows():
            review_text = row['review_text']
            if len(review_text) > 100:  #If the review is too long, just show the first 100 characters
                review_text = review_text[:100] + '...'  #Truncate text for brevity
            print(f"Review: {review_text}\nTrue Sentiment: {row['review_classification']} | Predicted Sentiment: {row['predicted_sentiment']}\n")
        
        #Save the misclassified reviews to a CSV for further analysis
        df_misclassified.to_csv("../../data/deep_learning/evaluation_results/dl_misclassified_reviews.csv", index=False)
        print("\nAll misclassified reviews saved to 'dl_misclassified_reviews.csv'")

    else:
        print("No misclassified reviews found.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except ValueError as e:
    print(f"Data error: {e}")
    exit()
