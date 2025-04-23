from afinn import Afinn
import pandas as pd

# Initialize AFINN
afinn = Afinn()

# Define keywords (adjusted for stronger influence)
positive_keywords = ['amazing', 'fantastic', 'excellent', 'perfect', 'outstanding', 'incredible']
negative_keywords = ['horrible', 'terrible', 'awful', 'worst', 'disastrous', 'poor']

# Refined neutral keywords with adjusted boosts
soft_neutral_keywords = ['okay', 'fine', 'average', 'decent', 'typical', 'standard', 'moderate', 'fair']
strong_neutral_keywords = ['neutral', 'acceptable', 'satisfactory', 'clean', 'on time', 'punctual', 'smooth']
neutral_phrases = ['nothing special', 'just fine', 'no big deal', 'nothing to complain about', 'as expected']

def analyze_sentiment(review_text):
    score = afinn.score(review_text)  # AFINN's word-level scoring
    lowered = review_text.lower()

    # 1. Apply stronger keyword boosts (word-level adjustments)
    for word in positive_keywords:
        if word in lowered:
            score += 1.1  # Slightly higher positive boost
    for word in negative_keywords:
        if word in lowered:
            score -= 1.1  # Slightly higher negative boost

    # 2. Neutral handling (adjusted boosts)
    neutral_boost = 0
    for word in strong_neutral_keywords:
        if word in lowered:
            neutral_boost -= 0.3  # More balanced penalty for strong neutral words
    for word in soft_neutral_keywords:
        if word in lowered:
            neutral_boost -= 0.1  # Lighter penalty for soft neutral words
    for phrase in neutral_phrases:
        if phrase in lowered:
            neutral_boost -= abs(score) * 0.6  # Adjust to lessen strong pull to neutral

    score += neutral_boost  # Apply neutral adjustments after scoring

    # 3. Classification thresholds (refined for better separation)
    if score > 2.0:
        sentiment = "Positive"
    elif score < -0.9:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, score

df = pd.read_csv("../../data/dataset/utest_set.csv")

# Apply sentiment analysis
df["review_classification"], df["confidence_score"] = zip(*df["review_text"].apply(analyze_sentiment))

# Save results
df.to_csv("../../data/wordscoring/t-wd_results.csv", index=False)

print("Word Scoring Sentiment analysis done")
