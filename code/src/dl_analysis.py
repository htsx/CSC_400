import pandas as pd
from transformers import pipeline

# Load the dataset
reviews_df = pd.read_csv('data/cleaned_skytrax_reviews.csv').head(20)  # Limit to 20 rows for testing

# Load a pre-trained sentiment analysis model
# This model is BERT-based and fine-tuned for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Perform sentiment analysis on each review
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Apply sentiment analysis to the dataset
reviews_df[['deep_learning_sentiment', 'deep_learning_score']] = reviews_df['review_text'].apply(
    lambda text: pd.Series(analyze_sentiment(str(text)))
)

# Save the results to a new CSV file
reviews_df.to_csv('data/deep_learning_sentiment.csv', index=False)

print("Deep learning sentiment analysis complete. Data saved to 'data/deep_learning_sentiment.csv'.")