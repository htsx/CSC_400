import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load the cleaned dataset
df = pd.read_csv("../../data/webscrapper/cleaned_skytrax_reviews.csv")

# Initialize VADER Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis using VADER
def vader_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    score = vader_analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Sentiment analysis using TextBlob
def textblob_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply both sentiment analyzers
df["VADER_Sentiment"] = df["review_text"].apply(vader_sentiment)
df["TextBlob_Sentiment"] = df["review_text"].apply(textblob_sentiment)

# Save results
df.to_csv("../../data/scoring_distribution/vanilla/sd_results.csv", index=False)
print("✅ Sentiment analysis completed. Results saved to data/scoring_distribution/vanilla/sd_results.csv.")
