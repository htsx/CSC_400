import pandas as pd
from transformers import pipeline

# This is a pre-trained model that can tell if text is positive or negative
sentiment_checker = pipeline("sentiment-analysis", model="textattack/roberta-base-imdb")

# Words that should be considered neutral
neutral_words = ["ok", "meh", "decent"]

# Function to check if text contains neutral words
def has_neutral_words(text):
    text_lower = text.lower()
    for word in neutral_words:
        if word in text_lower:
            return True
    return False

# Function to analyze the sentiment of a review
def get_sentiment(review_text):
    try:
        # First check for neutral words
        if has_neutral_words(review_text):
            return "NEUTRAL", 0.99, {'POSITIVE': 0.0, 'NEUTRAL': 0.99, 'NEGATIVE': 0.0}
        
        # Get sentiment from the model
        result = sentiment_checker(review_text[:1000])[0]  # Using first 1000 characters to avoid errors
        
        sentiment = result['label'].upper()
        confidence = round(result['score'], 2)
        
        # If confidence isn't strong, consider it neutral
        if confidence < 0.6:
            sentiment = "NEUTRAL"
        
        # Prepare scores for all categories
        scores = {
            'POSITIVE': 0.0,
            'NEUTRAL': 0.0,
            'NEGATIVE': 0.0
        }
        scores[sentiment] = confidence
        
        return sentiment, confidence, scores
    
    except Exception as e:
        print(f"Couldn't analyze this review because: {e}")
        return None, None, None

# Load the reviews data
reviews_data = pd.read_csv("../../data/webscrapper/cleaned_skytrax_reviews.csv")

# Make sure we have the right column
if 'review_text' not in reviews_data.columns:
    print("Error: The CSV file needs a 'review_text' column.")
    exit()

# Work with only the first 1000 reviews to save time
reviews_data = reviews_data.head(1000)

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

# Save the results to a new CSV file
results_df = pd.DataFrame(analysis_results)
results_df.to_csv("../../data/deep_learning/dl_results.csv", index=False)

print("Analysis complete! Results saved to dl_results.csv")