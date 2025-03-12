import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#load the reviews (20 for example)
reviews_df = pd.read_csv('data/cleaned_skytrax_reviews.csv').head(20)

#calculate sentiment scores
def sentiment_scoring(text):
    textblob_score = TextBlob(text).sentiment.polarity
    vader_score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    return textblob_score, vader_score

#apply the sentiment scoring to the reviews
reviews_df[['textblob_sentiment', 'vader_sentiment']] = reviews_df['review_text'].apply(
    lambda text: pd.Series(sentiment_scoring(str(text)))
)

#calculate average sentiment
reviews_df['average_sentiment'] = (reviews_df['textblob_sentiment'] + reviews_df['vader_sentiment']) / 2

#classify the sentiment scoring
reviews_df['sentiment_label'] = reviews_df['average_sentiment'].apply(
    lambda score: 'positive' if score > 0.05 else ('neutral' if -0.05 <= score <= 0.05 else 'negative')
)

#round sentiment scores to 4 decimal places
reviews_df['textblob_sentiment'] = reviews_df['textblob_sentiment'].round(4)
reviews_df['vader_sentiment'] = reviews_df['vader_sentiment'].round(4)
reviews_df['average_sentiment'] = reviews_df['average_sentiment'].round(4)

#reorder csv columns
reviews_df = reviews_df[['textblob_sentiment', 'vader_sentiment', 'average_sentiment', 'sentiment_label'] + 
                        [col for col in reviews_df.columns if col not in ['textblob_sentiment', 'vader_sentiment', 'average_sentiment', 'sentiment_label']]]

#save the results
reviews_df.to_csv('data/sd_sentiment.csv', index=False)

print("Sentiment analysis complete. Data saved to 'data/sd_sentiment.csv'.")