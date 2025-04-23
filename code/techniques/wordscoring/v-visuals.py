import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the CSV files containing our evaluation data
classification_report_df = pd.read_csv('../../data/wordscoring/v-wd_classification_report.csv', index_col=0)
confusion_matrix_df = pd.read_csv('../../data/wordscoring/v-wd_confusion_matrix.csv', index_col=0)
evaluation_metrics_df = pd.read_csv('../../data/wordscoring/v-wd_evaluation_metrics.csv', header=0)

#Remove "accuracy" from the classification report if it’s there
classification_report_df = classification_report_df[~classification_report_df.index.str.contains("accuracy", case=False)]

#Create a heatmap to visualize the confusion matrix (how the model predicted vs actual labels)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_df.astype(int), annot=True, fmt='d', cmap='Blues', 
            xticklabels=confusion_matrix_df.columns, yticklabels=confusion_matrix_df.index)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()
plt.close()

#For each metric, we will plot a bar chart
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    if metric in classification_report_df.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=classification_report_df.index, y=classification_report_df[metric])
        plt.title(f'{metric.capitalize()} per Class')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()

#If we have these metrics, let’s plot them together in one grouped bar chart
available_metrics = [metric for metric in ['precision', 'recall', 'f1-score'] if metric in classification_report_df.columns]

if len(available_metrics) > 0:
    classification_report_df[available_metrics].plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title('Grouped Bar Chart of Precision, Recall, F1-Score per Class')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    plt.legend(title="Metrics")
    plt.tight_layout()
    plt.show()
else:
    print("No valid metrics found to plot.")

accuracy = evaluation_metrics_df['Accuracy'].values[0]
labels = ['Correct', 'Incorrect']
sizes = [accuracy, 1 - accuracy]
colors = ['#4CAF50', '#FF6F61']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', wedgeprops={'width': 0.4})
plt.title('Model Accuracy')
plt.tight_layout()
plt.show()
plt.close()
