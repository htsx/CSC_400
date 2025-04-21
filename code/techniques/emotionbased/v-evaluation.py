import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

#Load ground truth and predicted sentiment files
skytrax_dataset = "../../data/dataset/validation_set.csv"
predictions_file = "../../data/scoringdistribution/v-sd_results.csv"

try:
    #Load the ground truth data and drop any rows with NaN in 'review_classification'
    df_truth = pd.read_csv(skytrax_dataset).dropna(subset=['review_classification'])
    
    #Load the predictions from the scoring distribution (our model's output)
    df_pred = pd.read_csv(predictions_file).reset_index(drop=True)

    #Make sure the number of reviews match between the ground truth and predictions
    if len(df_truth) != len(df_pred):
        raise ValueError(f"Mismatch in the number of reviews: {len(df_truth)} Skytrax Dataset reviews, but {len(df_pred)} predicted reviews.")

    #Extract the actual and predicted sentiment labels and ensure they’re cleaned up (no extra spaces, capitalized)
    y_true = df_truth['review_classification'].astype(str).str.strip().str.capitalize()
    y_pred = df_pred['review_classification'].astype(str).str.strip().str.capitalize()

    #Define sentiment categories for our analysis
    sentiment_labels = ['Negative', 'Neutral', 'Positive']

    #Calculate the evaluation metrics, making sure we handle any zero division errors
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    #Display the evaluation metrics
    print("\nEvaluation Metrics for the reviews:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    #Save the evaluation metrics to a CSV file
    metrics_data = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv("../../data/scoringdistribution/v-sd_evaluation_metrics.csv", index=False)

    #Generate and save classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=sentiment_labels, zero_division=0)
    print(report)
    report_data = classification_report(y_true, y_pred, target_names=sentiment_labels, output_dict=True)
    df_report = pd.DataFrame(report_data).transpose()
    df_report.to_csv("../../data/scoringdistribution/v-sd_classification_report.csv", index=True)

    #Generate and save confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=sentiment_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    df_conf_matrix = pd.DataFrame(conf_matrix, index=sentiment_labels, columns=sentiment_labels)
    df_conf_matrix.to_csv("../../data/scoringdistribution/v-sd_confusion_matrix.csv", index=True)

    #Save misclassified reviews to CSV and print count
    df_misclassified = df_truth.copy()
    df_misclassified['predicted_classification'] = y_pred
    df_misclassified = df_misclassified[df_misclassified['review_classification'].str.strip().str.capitalize() != df_misclassified['predicted_classification']]
    
    num_misclassified = len(df_misclassified)
    print(f"\nNumber of misclassified reviews: {num_misclassified}")

    if not df_misclassified.empty:
        df_misclassified.to_csv("../../data/scoringdistribution/v-sd_misclassified_reviews.csv", index=False)

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except ValueError as e:
    print(f"Data error: {e}")
    exit()
