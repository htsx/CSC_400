import pandas as pd
from transformers import pipeline

# Load the cardiffnlp/twitter-xlm-roberta-base-sentiment model for sentiment analysis
sentiment_checker = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Words that should be considered neutral
neutral_words = ["ok", "meh", "decent"]

def has_neutral_words(text):
    """Check if the text contains any predefined neutral words."""
    text_lower = text.lower()
    return any(word in text_lower for word in neutral_words)

def get_sentiment(review_text):
    """Analyze the sentiment of a review using the pre-trained model."""
    try:
        if has_neutral_words(review_text):
            return "NEUTRAL", 0.99, {'POSITIVE': 0.0, 'NEUTRAL': 0.99, 'NEGATIVE': 0.0}
        
        # Get sentiment from the model
        result = sentiment_checker(review_text[:1000])[0]  # Still limiting input to 1000 chars for model stability
        sentiment = result['label'].upper()
        confidence = round(result['score'], 2)

        if confidence < 0.55:
            sentiment = "NEUTRAL"

        # Set up sentiment scores
        scores = {'POSITIVE': 0.0, 'NEUTRAL': 0.0, 'NEGATIVE': 0.0}
        scores[sentiment] = confidence

        return sentiment, confidence, scores
    except Exception as e:
        print(f"Error analyzing review: {e}")
        return None, None, None

# Load the reviews data
reviews_data = pd.read_csv("../../data/webscrapper/cleaned_skytrax_reviews.csv")

if 'review_text' not in reviews_data.columns:
    print("Error: The CSV file needs a 'review_text' column.")
    exit()

# Analyze each review and store results
analysis_results = []
for _, row in reviews_data.iterrows():
    sentiment, confidence, scores = get_sentiment(row['review_text'])

    analysis_results.append({
        'review_name': row['review_name'],
        'review_type': row['review_type'],
        'passenger_name': row['passenger_name'],
        'review_date': row['review_date'],
        'review_text': row['review_text'],
        'sentiment': sentiment,
        'confidence': confidence,
        'POS': scores['POSITIVE'] if scores else None,
        'NEU': scores['NEUTRAL'] if scores else None,
        'NEG': scores['NEGATIVE'] if scores else None
    })

# Save results
results_df = pd.DataFrame(analysis_results)
results_df.to_csv("../../data/deep_learning/dl_results.csv", index=False)

print("Analysis complete! Results saved to dl_results.csv")
