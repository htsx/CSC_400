import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(filepath):
    """Plots the distribution of sentiment scores."""
    data = pd.read_csv(filepath)
    sns.histplot(data['sentiment_score'], kde=True)
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    input_path = '../data/sentiment_reviews.csv'
    plot_sentiment_distribution(input_path)
