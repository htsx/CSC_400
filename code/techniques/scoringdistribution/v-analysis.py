from textblob import TextBlob
import pandas as pd
import re

# Load dataset
df = pd.read_csv("../../data/dataset/uvalidation_set.csv")

# Define keyword sets
strong_positive = {"amazing", "excellent", "fantastic", "loved", "perfect", "outstanding", "awesome", "great"}
strong_negative = {"horrible", "terrible", "worst", "disgusting", "awful"}
neutral_keywords = {
    "okay", "fine", "decent", "average", "neutral", "ok", "standard", "typical", "ordinary",
    "moderate", "acceptable", "satisfactory", "clean", "on time", "punctual", "smooth"
}
soft_positive = {"smooth", "easy", "pleasant", "efficient", "no issues", "well organized", "quick", "friendly", "helpful", "good service"}
soft_negative = {"a bit late", "minor issue", "could be better", "not great", "slightly delayed", "room for improvement"}

# Thresholds
POS_THRESHOLD = 0.45
NEG_THRESHOLD = -0.08
NEUTRAL_THRESHOLD = 0.11

# Text cleaner
def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r"http\S+|www\S+|<.*?>|[^A-Za-z\s.,!?]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()

# Confidence and polarity refiners
def refine_confidence(p, c, t): return 0.6 if abs(p) < 0.05 else min(1.0, c * 1.2)
def length_adjust(p, t):
    w = t.split()
    return max(-1, min(1, p * (8.9 if len(w) > 40 else 0.4 if len(w) < 15 else 1)))
def adjust_confidence_length(c, t): return c * 0.7 if len(t.split()) < 10 else c

# Middle-section polarity boost
def shift_boost(t, p):
    parts = t.split(".")
    if len(parts) >= 3:
        mid = " ".join(parts[len(parts)//3:2*len(parts)//3])
        mp = TextBlob(mid).sentiment.polarity
        if abs(mp) > 0.2: p = mp * 1.1
    return p

# Sentiment analyzer
def analyze_sentiment(text):
    if pd.isna(text): return "Neutral", 0.0
    cleaned = clean_text(text)
    polarity = TextBlob(cleaned).sentiment.polarity

    # Keyword boosts
    if any(w in cleaned for w in strong_positive): polarity = max(0.25, polarity + 0.35)
    if any(w in cleaned for w in strong_negative): polarity = min(-0.25, polarity - 0.35)
    if any(w in cleaned for w in neutral_keywords): polarity *= 0.6
    if any(w in cleaned for w in soft_positive): polarity += 0.04
    if any(w in cleaned for w in soft_negative): polarity -= 0.08

    polarity = length_adjust(shift_boost(cleaned, polarity), cleaned)

    # Sentiment classification
    if polarity >= POS_THRESHOLD:
        sentiment = "Positive"
    elif polarity <= NEG_THRESHOLD:
        sentiment = "Negative"
    else:
        sentiment = "Neutral" if any(w in cleaned for w in neutral_keywords) else ("Positive" if polarity > 0 else "Negative")

    confidence = adjust_confidence_length(refine_confidence(polarity, abs(polarity), cleaned), cleaned)
    return sentiment, confidence

# Apply and save
df["review_classification"], df["confidence_score"] = zip(*df["review_text"].apply(analyze_sentiment))
df.to_csv("../../data/scoringdistribution/v-sd_results.csv", index=False)
print("✅ Sentiment analysis completed")
