import matplotlib.pyplot as plt
import seaborn as sns

# Plot sentiment label distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='deep_learning_sentiment', data=reviews_df, palette='viridis')
plt.title('Deep Learning Sentiment Label Distribution')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.show()