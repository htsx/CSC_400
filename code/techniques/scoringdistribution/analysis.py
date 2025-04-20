from textblob import TextBlob
import pandas as pd
import re

# Load your review dataset
df = pd.read_csv("../../data/dataset/ubalanced.csv")

# Sentiment keyword sets
strong_positive = {"amazing", "excellent", "fantastic", "loved", "perfect", "outstanding", "awesome", "great"}
strong_negative = {"horrible", "terrible", "worst", "disgusting", "awful"}
neutral_keywords = {
    "okay", "fine", "decent", "average", "neutral", "ok", "standard",
    "typical", "ordinary", "moderate", "acceptable", "satisfactory",
    "clean", "on time", "punctual", "smooth"
}
soft_positive = {"smooth", "easy", "pleasant", "efficient", "no issues", "well organized", "quick", "friendly", "helpful", "good service"}
soft_negative = {"a bit late", "minor issue", "could be better", "not great", "slightly delayed", "room for improvement"}
soft_neutral = {"nothing special", "just fine", "it was ok", "no big deal", "neither good nor bad", "as expected", "typical experience", "average at best", "nothing to complain about"}

# Clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Optimized thresholds - reverted to original working values
POS_THRESHOLD = 0.32
NEG_THRESHOLD = -0.12
NEUTRAL_THRESHOLD = 0.11  # Expanded neutral threshold for more flexibility

# Confidence adjustment - simplified
def refine_confidence(polarity, confidence, text):
    abs_polarity = abs(polarity)
    if abs_polarity < 0.05:
        return 0.6  # Low confidence for neutral cases
    return min(1.0, confidence * 1.2)  # Boost for clear cases

# Length-based adjustment - kept original
def length_based_adjustment(polarity, text):
    words = text.split()
    if len(words) > 40:
        polarity *= 8.9
    elif len(words) < 15:
        polarity *= 0.4
    return max(-1, min(1, polarity))

def normalize_confidence_for_length(confidence, text):
    if len(text.split()) < 10:
        return confidence * 0.7
    return confidence

def sentiment_shift_boost(text, polarity):
    parts = text.split('.')
    if len(parts) >= 3:
        middle = ' '.join(parts[len(parts)//3:2*len(parts)//3])
        middle_blob = TextBlob(middle)
        if abs(middle_blob.sentiment.polarity) > 0.2:
            polarity = middle_blob.sentiment.polarity * 1.1
    return polarity

def textblob_sentiment_with_triggers(text):
    if pd.isna(text):
        return "Neutral", 0.0

    cleaned_text = clean_text(text)
    blob = TextBlob(cleaned_text)
    polarity = blob.sentiment.polarity

    # Keyword adjustments - rebalanced with safeguards
    if any(w in cleaned_text for w in strong_positive):
        polarity = max(0.25, polarity + 0.35)  # Don't let strong positives go negative
    if any(w in cleaned_text for w in strong_negative):
        polarity = min(-0.25, polarity - 0.35)  # Don't let strong negatives go positive
    if any(w in cleaned_text for w in neutral_keywords):
        polarity *= 0.6  # Boost neutral keyword influence
    if any(w in cleaned_text for w in soft_positive):
        polarity += 0.04  # Original soft positive boost
    if any(w in cleaned_text for w in soft_negative):
        polarity -= 0.08  # Original soft negative penalty

    polarity = sentiment_shift_boost(cleaned_text, polarity)
    polarity = length_based_adjustment(polarity, cleaned_text)
    polarity = max(-1, min(1, polarity))

    # Classification - updated neutral logic
    if polarity >= POS_THRESHOLD:
        sentiment = "Positive"
    elif polarity <= NEG_THRESHOLD:
        sentiment = "Negative"
    else:
        # Apply expanded neutral threshold logic
        if (polarity > NEG_THRESHOLD and polarity < POS_THRESHOLD) or \
           any(w in cleaned_text for w in neutral_keywords):
            sentiment = "Neutral"
        else:
            # For values between thresholds but without neutral keywords
            sentiment = "Positive" if polarity > 0 else "Negative"
    

    confidence = abs(polarity)
    adjusted_confidence = refine_confidence(polarity, confidence, cleaned_text)
    adjusted_confidence = normalize_confidence_for_length(adjusted_confidence, cleaned_text)

    return sentiment, adjusted_confidence

# Apply to all reviews
df["review_classification"], df["confidence_score"] = zip(*df["review_text"].apply(textblob_sentiment_with_triggers))

# Save
df.to_csv("../../data/scoringdistribution/sd_results.csv", index=False)
print("✅ Sentiment analysis completed")
