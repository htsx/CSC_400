import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_skytrax_reviews.csv")

# Initialize VADER Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment using VADER
def vader_sentiment(text):
    if pd.isna(text):  # Handle missing values
        return "Neutral"
    score = vader_analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to analyze sentiment using TextBlob
def textblob_sentiment(text):
    if pd.isna(text):  # Handle missing values
        return "Neutral"
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["VADER_Sentiment"] = df["review_text"].apply(vader_sentiment)
df["TextBlob_Sentiment"] = df["review_text"].apply(textblob_sentiment)


# Save results
df.to_csv("data/sd_results.csv", index=False)
print("✅ Sentiment analysis completed. Results saved to data/sentiment_results.csv.")

