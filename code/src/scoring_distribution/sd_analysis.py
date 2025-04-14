from textblob import TextBlob
import pandas as pd
import re

# Load your review dataset
df = pd.read_csv("../../data/webscrapper/unlabeled_reviews.csv")

# Strong sentiment keywords
strong_positive = {"amazing", "excellent", "fantastic", "loved", "perfect"}
strong_negative = {"horrible", "terrible", "worst", "disgusting", "awful"}

# Clean the text before sentiment analysis
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)          # Remove HTML tags
    text = re.sub(r'[^A-Za-z\s.,!?]', '', text)  # Keep only allowed characters
    text = re.sub(r'\s+', ' ', text).strip()   # Normalize whitespace
    return text.lower()

# 🧠 Adjusted thresholds for better classification balance
POS_THRESHOLD = 0.10      # Increased from 0.02 (requires stronger positive signal)
NEG_THRESHOLD = -0.10     # Increased from -0.01 (requires stronger negative signal)

# 🔍 Enhanced confidence refinement
def refine_confidence(polarity, confidence):
    """More sophisticated confidence adjustment"""
    abs_polarity = abs(polarity)
    
    if abs_polarity < 0.05:  # Very close to neutral
        return 0.7  # High confidence neutral
    elif abs_polarity < 0.2:  # Weak signal
        return confidence * 0.6
    else:  # Strong signal
        return min(1.0, confidence * 1.2)

def textblob_sentiment_with_triggers(text):
    if pd.isna(text):
        return "Neutral", 0.0

    cleaned_text = clean_text(text)
    blob = TextBlob(cleaned_text)
    polarity = blob.sentiment.polarity
    
    # Check for strong sentiment keywords
    positive_trigger = any(word in cleaned_text.split() for word in strong_positive)
    negative_trigger = any(word in cleaned_text.split() for word in strong_negative)

    # Adjust sentiment based on strong keywords
    if positive_trigger:
        sentiment = "Positive"
        confidence = 1.0  # Maximum confidence for strong positive sentiment
    elif negative_trigger:
        sentiment = "Negative"
        confidence = 1.0  # Maximum confidence for strong negative sentiment
    else:
        # Sentiment assignment based on polarity and refined thresholds
        if polarity >= POS_THRESHOLD:
            sentiment = "Positive"
        elif polarity <= NEG_THRESHOLD:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        confidence = abs(polarity)  # Confidence is the absolute polarity strength

    adjusted_confidence = refine_confidence(polarity, confidence)

    return sentiment, adjusted_confidence

# 🔁 Apply sentiment classification across the dataset
df["review_classification"], df["confidence_score"] = zip(*df["review_text"].apply(textblob_sentiment_with_triggers))

# 💾 Save your results for evaluation
df.to_csv("../../data/scoring_distribution/tweaked/sd_results.csv", index=False)
print("✅ Sentiment analysis completed with strong keyword triggers and refined thresholds.")
