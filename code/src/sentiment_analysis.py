import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the first 20 reviews
reviews_df = pd.read_csv('data/cleaned_skytrax_reviews.csv').head(20)

# Calculate sentiment scores
def sentiment_scoring(text):
    textblob_score = TextBlob(text).sentiment.polarity
    vader_score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    return textblob_score, vader_score

# Apply sentiment scoring
reviews_df[['textblob_sentiment', 'vader_sentiment']] = reviews_df['review_text'].apply(
    lambda text: pd.Series(sentiment_scoring(str(text)))
)

# Calculate average sentiment
reviews_df['average_sentiment'] = (reviews_df['textblob_sentiment'] + reviews_df['vader_sentiment']) / 2

# Classify sentiment
reviews_df['sentiment_label'] = reviews_df['average_sentiment'].apply(
    lambda score: 'positive' if score > 0.05 else ('neutral' if -0.05 <= score <= 0.05 else 'negative')
)

# Reorder columns
reviews_df = reviews_df[['textblob_sentiment', 'vader_sentiment', 'average_sentiment', 'sentiment_label'] + 
                        [col for col in reviews_df.columns if col not in ['textblob_sentiment', 'vader_sentiment', 'average_sentiment', 'sentiment_label']]]

# Save results
reviews_df.to_csv('data/sentiment_scoring_reviews.csv', index=False)

print("Sentiment analysis complete. Data saved to 'data/sentiment_scoring_reviews.csv'.")