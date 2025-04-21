import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the CSV files (placeholders for the data)
classification_report_df = pd.read_csv('../../data/keywordstopics/t_kt_classification_report.csv', index_col=0)
confusion_matrix_df = pd.read_csv('../../data/keywordstopics/t_kt_confusion_matrix.csv', index_col=0)
evaluation_metrics_df = pd.read_csv('../../data/keywordstopics/t_kt_evaluation_metrics.csv', header=0)

#Remove "accuracy" from the classification report if it’s there (we don’t need it for this analysis)
classification_report_df = classification_report_df[~classification_report_df.index.str.contains("accuracy", case=False)]

plt.figure(figsize=(8, 6))  #Setting up the figure size for the confusion matrix
sns.heatmap(confusion_matrix_df.astype(int), annot=True, fmt='d', cmap='Blues', 
            xticklabels=confusion_matrix_df.columns, yticklabels=confusion_matrix_df.index)  #Plotting the heatmap
plt.title('Confusion Matrix')  #Title for the heatmap
plt.xlabel('Predicted Labels')  #Label for the X-axis
plt.ylabel('True Labels')  #Label for the Y-axis
plt.tight_layout()  #Making sure everything fits nicely
plt.show()  #Show the plot
plt.close()  #Close the plot to free up memory

metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    if metric in classification_report_df.columns:  #Check if the metric is available in the report
        plt.figure(figsize=(8, 6))  #Setting up the figure size for each plot
        sns.barplot(x=classification_report_df.index, y=classification_report_df[metric])  #Plot the bar chart
        plt.title(f'{metric.capitalize()} per Class')  #Title based on the metric
        plt.ylabel(metric.capitalize())  #Y-axis label is the metric (Precision, Recall, or F1-score)
        plt.xlabel('Class')  #X-axis label is the class
        plt.xticks(rotation=45)  #Rotate the X-axis labels for better readability
        plt.tight_layout()  #Ensure the layout doesn’t overlap
        plt.show()  #Show the plot
        plt.close()  #Close the plot to free up memory

#Check if we have valid metrics to plot (just in case something’s missing)
available_metrics = [metric for metric in ['precision', 'recall', 'f1-score'] if metric in classification_report_df.columns]

if len(available_metrics) > 0:  #Only plot if we have at least one valid metric
    classification_report_df[available_metrics].plot(kind='bar', figsize=(10, 6), width=0.8)  #Create a grouped bar chart
    plt.title('Grouped Bar Chart of Precision, Recall, F1-Score per Class')  #Chart title
    plt.ylabel('Score')  #Y-axis label
    plt.xlabel('Class')  #X-axis label
    plt.xticks(rotation=45)  #Rotate the X-axis labels for clarity
    plt.legend(title="Metrics")  #Add a legend to show which color represents which metric
    plt.tight_layout()  #Make sure everything is properly aligned
    plt.show()  #Show the plot
else:
    print("No valid metrics found to plot.")  #Print a message if no valid metrics are available

#Get the accuracy value, set to 0 if it's not found in the metrics file
accuracy = evaluation_metrics_df['Accuracy'].values[0] if 'Accuracy' in evaluation_metrics_df.columns else 0
labels = ['Correct', 'Incorrect']  #Labels for the donut chart
sizes = [accuracy, 1 - accuracy]  #Sizes based on accuracy and the rest
colors = ['#4CAF50', '#FF6F61']  #Color scheme for the chart

plt.figure(figsize=(6, 6))  # et the figure size for the donut chart
plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', wedgeprops={'width': 0.4})  #Plot the donut chart
plt.title('Model Accuracy')  #Title of the donut chart
plt.tight_layout()  #Make sure everything fits
plt.show()  #Display the chart
plt.close()  #Close the plot to free up memory
