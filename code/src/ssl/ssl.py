import pandas as pd
from transformers import pipeline
import numpy as np

# Load the cardiffnlp/twitter-xlm-roberta-base-sentiment model for sentiment analysis
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

# Load the labeled reviews data
labeled_data = pd.read_csv("../../data/webscrapper/cleaned_skytrax_reviews.csv")

# Identify labeled data (has ground truth sentiment)
labeled_data = labeled_data.dropna(subset=['sentiment'])  # Keep only labeled data

# Load the unlabeled data (optional: if you have a separate file)
unlabeled_data = pd.read_csv("../../data/webscrapper/unlabeled_reviews.csv")  # Example: Unlabeled reviews

# Pseudo-labeling: Get predictions for unlabeled data
pseudo_labels = []
for _, row in unlabeled_data.iterrows():
    sentiment, confidence = get_sentiment(row['review_text'])
    if confidence > 0.7:  # Only accept high-confidence predictions
        pseudo_labels.append({'review_text': row['review_text'], 'predicted_sentiment': sentiment, 'confidence': confidence})

# Create a DataFrame of pseudo-labeled data
pseudo_labeled_df = pd.DataFrame(pseudo_labels)

# Combine labeled data with pseudo-labeled data
combined_data = pd.concat([labeled_data[['review_text', 'sentiment']], pseudo_labeled_df[['review_text', 'predicted_sentiment']]])

# Optionally, you can shuffle the combined data before retraining
combined_data = combined_data.sample(frac=1, random_state=42)

# Retrain your model using the combined dataset (This can be done with your model training loop)
# Here, you would ideally reformat the data to fit the model's training procedure.

print("Pseudo-labeling complete! Combined dataset has been generated.")
