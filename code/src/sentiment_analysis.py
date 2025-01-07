from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment(text):
    """Performs sentiment analysis on a single text."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def add_sentiment_scores(filepath):
    """Adds sentiment scores to the dataset."""
    data = pd.read_csv(filepath)
    data['sentiment_score'] = data['cleaned_text'].apply(analyze_sentiment)
    return data

if __name__ == "__main__":
    input_path = '../data/cleaned_reviews.csv'
    output_path = '../data/sentiment_reviews.csv'
    sentiment_data = add_sentiment_scores(input_path)
    sentiment_data.to_csv(output_path, index=False)
