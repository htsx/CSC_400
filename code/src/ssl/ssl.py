import pandas as pd
from transformers import pipeline

# Load the sentiment analysis model
sentiment_checker = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Function to get sentiment prediction
def get_sentiment(review_text):
    try:
        result = sentiment_checker(review_text[:1000])[0]  # Limit to 1000 characters
        sentiment = result['label'].upper()
        confidence = result['score']
        return sentiment, confidence
    except Exception as e:
        print(f"Error analyzing review: {e}")
        return None, None

# Load labeled reviews
labeled_data = pd.read_csv("../../data/webscrapper/labeled_reviews.csv")

# Load unlabeled reviews
unlabeled_data = pd.read_csv("../../data/webscrapper/unlabeled_reviews.csv")

# DataFrames to store pseudo-labeled and manual check reviews
pseudo_labels = []
manual_labeling = []

# Pseudo-labeling process
for _, row in unlabeled_data.iterrows():
    sentiment, confidence = get_sentiment(row['review_text'])
    
    if confidence is not None:
        if confidence >= 0.90:
            pseudo_labels.append({'review_text': row['review_text'], 'sentiment': sentiment, 'confidence': confidence})
        else:
            manual_labeling.append({'review_text': row['review_text'], 'confidence': confidence})

# Convert to DataFrames
pseudo_labeled_df = pd.DataFrame(pseudo_labels)
manual_labeling_df = pd.DataFrame(manual_labeling)

# Save pseudo-labeled and manual labeling data
pseudo_labeled_df.to_csv("../../data/deep_learning/pseudo_labeled_reviews.csv", index=False)
manual_labeling_df.to_csv("../../data/deep_learning/manual_labeling_reviews.csv", index=False)

# Combine labeled data with high-confidence pseudo-labeled data
combined_data = pd.concat([labeled_data[['review_text', 'sentiment']], pseudo_labeled_df[['review_text', 'sentiment']]])

# Shuffle dataset before training
combined_data = combined_data.sample(frac=1, random_state=42)

# Save final dataset for training
combined_data.to_csv("../../data/deep_learning/ssl_training_dataset.csv", index=False)

print("Pseudo-labeling complete! Datasets saved:")
print("- High-confidence pseudo-labeled reviews → pseudo_labeled_reviews.csv")
print("- Low-confidence reviews for manual check → manual_labeling_reviews.csv")
print("- Final SSL training dataset → ssl_training_dataset.csv")
