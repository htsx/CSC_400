import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the CSV files with the evaluation results
classification_report_df = pd.read_csv('../../data/deeplearning/dl_classification_report.csv', index_col=0)
confusion_matrix_df = pd.read_csv('../../data/deeplearning/dl_confusion_matrix.csv', index_col=0)
evaluation_metrics_df = pd.read_csv('../../data/deeplearning/dl_evaluation_metrics.csv', header=0)

#Remove "accuracy" from the classification report if it’s there (we don’t need it here)
classification_report_df = classification_report_df[~classification_report_df.index.str.contains("accuracy", case=False)]

#Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
#Plot the confusion matrix using a heatmap for better visualization
sns.heatmap(confusion_matrix_df.astype(int), annot=True, fmt='d', cmap='Blues', 
            xticklabels=confusion_matrix_df.columns, yticklabels=confusion_matrix_df.index)
plt.title('Confusion Matrix')  #Set the title of the heatmap
plt.xlabel('Predicted Labels')  #Label the x-axis
plt.ylabel('True Labels')  #Label the y-axis
plt.tight_layout()  #Ensure everything fits nicely
plt.show()  #Show the plot
plt.close()  #Close the plot after showing it

#Separate Bar Plots for Precision, Recall, F1-Score
metrics = ['precision', 'recall', 'f1-score']  #List of metrics we want to plot
for metric in metrics:
    if metric in classification_report_df.columns:
        plt.figure(figsize=(8, 6))
        #For each metric, plot a bar chart to visualize the scores per class
        sns.barplot(x=classification_report_df.index, y=classification_report_df[metric])
        plt.title(f'{metric.capitalize()} per Class')  #Title with the metric name
        plt.ylabel(metric.capitalize())  #Label y-axis with the metric name
        plt.xlabel('Class')  #Label x-axis with 'Class'
        plt.xticks(rotation=45)  #Rotate the x-tick labels for better readability
        plt.tight_layout()  #Make sure the plot layout looks good
        plt.show()  #Show the plot
        plt.close()  #Close the plot after showing it

#Grouped Bar Chart for All Metrics (Precision, Recall, F1-Score)
plt.figure(figsize=(10, 6))

#Grouped Bar Chart for All Metrics (Precision, Recall, F1-Score)
#Check which metrics are available in the classification report and plot them
available_metrics = [metric for metric in ['precision', 'recall', 'f1-score'] if metric in classification_report_df.columns]

if len(available_metrics) > 0:
    #Plot the grouped bar chart if the metrics are available
    classification_report_df[available_metrics].plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title('Grouped Bar Chart of Precision, Recall, F1-Score per Class')  #Title for the chart
    plt.ylabel('Score')  #Y-axis label for the score
    plt.xlabel('Class')  #X-axis label for the class
    plt.xticks(rotation=45)  #Rotate the x-tick labels
    plt.legend(title="Metrics")  #Legend for the metrics
    plt.tight_layout()  #Ensure the layout is clean
    plt.show()  #Show the plot
else:
    print("No valid metrics found to plot.")  #If there are no valid metrics to plot, print a message

#Donut Chart for Accuracy
accuracy = evaluation_metrics_df['Accuracy'].values[0]  #Get the accuracy score
labels = ['Correct', 'Incorrect']  #Labels for the donut chart
sizes = [accuracy, 1 - accuracy]  #Sizes for the donut chart
colors = ['#4CAF50', '#FF6F61']  #Colors for the chart (green for correct, red for incorrect)

plt.figure(figsize=(6, 6))
#Plot the donut chart to show the model accuracy
plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', wedgeprops={'width': 0.4})
plt.title('Model Accuracy')  #Title for the chart
plt.tight_layout()  #Make sure the layout fits well
plt.show()  #Show the plot
plt.close()  #Close the plot after showing it
