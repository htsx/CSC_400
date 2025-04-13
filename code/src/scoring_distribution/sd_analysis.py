import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load cleaned reviews
df = pd.read_csv("../../data/webscrapper/cleaned_skytrax_reviews.csv")

# Initialize analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# VADER scoring
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

# TextBlob scoring
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

# Keyword-based fallback (basic rule-based sentiment booster)
def keyword_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    text = str(text).lower()
    positive_keywords = ["excellent", "amazing", "perfect", "great", "comfortable", "pleasant", "friendly"]
    negative_keywords = ["terrible", "awful", "worst", "unfriendly", "dirty", "late", "rude", "cancelled"]

    pos_hits = any(word in text for word in positive_keywords)
    neg_hits = any(word in text for word in negative_keywords)

    if pos_hits and not neg_hits:
        return "Positive"
    elif neg_hits and not pos_hits:
        return "Negative"
    elif pos_hits and neg_hits:
        return "Neutral"
    else:
        return "Neutral"

# Combine all three using hybrid rule-based logic
def hybrid_sentiment(text):
    vader = vader_sentiment(text)
    textblob = textblob_sentiment(text)
    keyword = keyword_sentiment(text)

    # Agreement between any two
    if vader == textblob:
        return vader
    elif vader == keyword:
        return vader
    elif textblob == keyword:
        return textblob
    else:
        return "Neutral"

# Apply all techniques
df["VADER_Sentiment"] = df["review_text"].apply(vader_sentiment)
df["TextBlob_Sentiment"] = df["review_text"].apply(textblob_sentiment)
df["Keyword_Sentiment"] = df["review_text"].apply(keyword_sentiment)
df["Hybrid_Sentiment"] = df["review_text"].apply(hybrid_sentiment)

# Save updated results
df.to_csv("../../data/scoring_distribution/sd_results.csv", index=False)
print("✅ Hybrid sentiment analysis completed. Results saved to data/scoring_distribution/sd_results.csv.")
