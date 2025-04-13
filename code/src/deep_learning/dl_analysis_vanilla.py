import pandas as pd
from transformers import pipeline

# Load the sentiment analysis model for Twitter-based sentiment classification
sentiment_checker = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# List of neutral words like "ok", "meh", "decent"
neutral_words = ["ok", "meh", "decent"]

def has_neutral_words(text):
    text_lower = text.lower()
    return any(word in text_lower for word in neutral_words)

def get_sentiment(review_text):
    try:
        if not isinstance(review_text, str) or review_text.strip() == "":
            return None, None, None

        result = sentiment_checker(review_text[:1000])[0]  
        sentiment = result['label'].upper()
        confidence = round(result['score'], 2)

        if confidence < 0.55:
            sentiment = "NEUTRAL"

        scores = {'POSITIVE': 0.0, 'NEUTRAL': 0.0, 'NEGATIVE': 0.0}
        scores[sentiment] = confidence

        return sentiment, confidence, scores
    except Exception as e:
        print(f"Error analyzing review: {e}")
        return None, None, None

# Load and clean data
reviews_data = pd.read_csv("../../data/webscrapper/unlabeled_reviews.csv")
reviews_data = reviews_data[reviews_data['review_text'].notna()]
reviews_data = reviews_data[reviews_data['review_text'].astype(str).str.strip() != ""]
reviews_data['review_text'] = reviews_data['review_text'].astype(str)

# Limit the data
reviews_data = reviews_data.head(12001).reset_index(drop=True)

# Filter out neutral reviews upfront
reviews_data['is_neutral'] = reviews_data['review_text'].apply(has_neutral_words)
neutral_reviews = reviews_data[reviews_data['is_neutral']]
non_neutral_reviews = reviews_data[~reviews_data['is_neutral']]

analysis_results = []

# Process neutral reviews
for _, row in neutral_reviews.iterrows():
    analysis_results.append({
        'review_name': row['review_name'],
        'review_type': row['review_type'],
        'passenger_name': row['passenger_name'],
        'review_date': row['review_date'],
        'review_text': row['review_text'],
        'sentiment': "NEUTRAL",
        'confidence': 0.99,
        'POS': 0.0,
        'NEU': 0.99,
        'NEG': 0.0
    })

# Process non-neutral reviews in batches
batch_size = 32  # You can increase this depending on memory and performance
for start in range(0, len(non_neutral_reviews), batch_size):
    batch = non_neutral_reviews.iloc[start:start + batch_size]
    texts = batch['review_text'].apply(lambda x: x[:1000]).tolist()

    try:
        predictions = sentiment_checker(texts)
        for i, result in enumerate(predictions):
            row = batch.iloc[i]
            sentiment = result['label'].upper()
            confidence = round(result['score'], 2)
            if confidence < 0.55:
                sentiment = "NEUTRAL"

            scores = {'POSITIVE': 0.0, 'NEUTRAL': 0.0, 'NEGATIVE': 0.0}
            scores[sentiment] = confidence

            analysis_results.append({
                'review_name': row['review_name'],
                'review_type': row['review_type'],
                'passenger_name': row['passenger_name'],
                'review_date': row['review_date'],
                'review_text': row['review_text'],
                'sentiment': sentiment,
                'confidence': confidence,
                'POS': scores['POSITIVE'],
                'NEU': scores['NEUTRAL'],
                'NEG': scores['NEGATIVE']
            })
    except Exception as e:
        print(f"Batch error: {e}")

# Save results
results_df = pd.DataFrame(analysis_results)
results_df.to_csv("../../data/deep_learning/vanilla/dl_results.csv", index=False)

print(f"Analysis complete! {len(results_df)} results saved to dl_results.csv")
