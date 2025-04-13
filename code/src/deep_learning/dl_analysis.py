from transformers import pipeline
import pandas as pd
import re


# Initialize the sentiment analysis pipeline
sentiment_checker = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


# Define label mapping
label_mapping = {
    'LABEL_0': 'NEGATIVE',
    'LABEL_1': 'NEUTRAL',
    'LABEL_2': 'POSITIVE'
}


# Define strong sentiment keywords
strong_positive = {"amazing", "excellent", "fantastic", "loved", "perfect"}
strong_negative = {"horrible", "terrible", "worst", "disgusting", "awful"}


def clean_text(text):
    """Basic cleaning of the text data."""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)  # Keep alphanumeric and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()


def adjust_threshold(sentiment, confidence,
                     positive_threshold=0.70,
                     negative_threshold=0.82,
                     neutral_threshold=0.75):
    """Adjust sentiment classification based on confidence thresholds."""
    if sentiment == 'POSITIVE' and confidence < positive_threshold:
        return 'NEUTRAL', 0.5
    elif sentiment == 'NEGATIVE' and confidence < negative_threshold:
        return 'NEUTRAL', 0.5
    elif sentiment == 'NEUTRAL' and confidence < neutral_threshold:
        return 'NEUTRAL', max(confidence, 0.75)
    return sentiment, confidence


def keyword_override(text, sentiment, confidence, threshold=0.60):
    """Override model prediction if strong sentiment keywords are found with low confidence."""
    tokens = text.lower().split()
    if confidence < threshold:
        if any(word in tokens for word in strong_positive):
            return "POSITIVE", 0.75
        elif any(word in tokens for word in strong_negative):
            return "NEGATIVE", 0.75
    return sentiment, confidence


def analyze_sentiment(text):
    """Analyze sentiment and return the predicted sentiment and confidence."""
    cleaned_text = clean_text(text)
    result = sentiment_checker(cleaned_text)[0]


    sentiment = label_mapping.get(result['label'], 'NEUTRAL')
    confidence = round(result['score'], 2)


    # Apply threshold and keyword logic
    sentiment, confidence = adjust_threshold(sentiment, confidence)
    sentiment, confidence = keyword_override(cleaned_text, sentiment, confidence)


    return sentiment.capitalize(), confidence


def process_reviews(df):
    """Process the reviews in the dataframe."""
    results = []
    for _, row in df.iterrows():
        sentiment, confidence = analyze_sentiment(row['review_text'])
        results.append({
            'review_name': row['review_name'],
            'review_type': row['review_type'],
            'passenger_name': row['passenger_name'],
            'review_date': row['review_date'],
            'review_text': row['review_text'],
            'sentiment': sentiment,
            'confidence': confidence
        })
    return pd.DataFrame(results)


# Load and process data
reviews_data = pd.read_csv("../../data/webscrapper/unlabeled_reviews.csv")
reviews_data = reviews_data[reviews_data['review_text'].notna()]
reviews_data['review_text'] = reviews_data['review_text'].astype(str)


results_df = process_reviews(reviews_data)
results_df.to_csv("../../data/deep_learning/tweaked/dl_results.csv", index=False)


print(f"Analysis complete! {len(results_df)} reviews processed.")
