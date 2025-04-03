import pandas as pd
from transformers import pipeline

# Load the sentiment analysis model from the local directory
sentiment_checker = pipeline("sentiment-analysis", model="./model")  # Update to your model directory

# Load labeled reviews
labeled_data = pd.read_csv("../../data/ground_truth/labeled_reviews.csv")

# DataFrames to store pseudo-labeled and manual check reviews
pseudo_labels = []
manual_labeling = []

# Print column names of labeled data
print("Labeled Data Columns:", labeled_data.columns)

# Function to get sentiment prediction
def get_sentiment(review_text):
    try:
        result = sentiment_checker(review_text[:1000])  # Limit to 1000 characters
        sentiment = result[0]['label'].upper() if result else None
        confidence = result[0]['score'] if result else None
        return sentiment, confidence
    except Exception as e:
        print(f"Error analyzing review: {e}")
        return None, None

# Load unlabeled reviews
unlabeled_data = pd.read_csv("../../data/ground_truth/unlabeled_reviews.csv")

# Pseudo-labeling process
for _, row in unlabeled_data.iterrows():
    sentiment, confidence = get_sentiment(row['review_text'])
    
    # Debugging output for each review being processed
    print(f"Review: {row['review_text'][:100]}...")  # Display the first 100 characters of the review
    print(f"Sentiment: {sentiment}, Confidence: {confidence}")
    
    if confidence is not None:
        if confidence >= 0.90:
            pseudo_labels.append({'review_text': row['review_text'], 'sentiment': sentiment, 'confidence': confidence})
        else:
            manual_labeling.append({'review_text': row['review_text'], 'confidence': confidence})

# Convert to DataFrames
pseudo_labeled_df = pd.DataFrame(pseudo_labels)
manual_labeling_df = pd.DataFrame(manual_labeling)

# Print column names of pseudo-labeled data
print("Pseudo-labeled Data Columns:", pseudo_labeled_df.columns)

# Save pseudo-labeled and manual labeling data
pseudo_labeled_df.to_csv("data/pseudo_labeled_reviews.csv", index=False)
manual_labeling_df.to_csv("data/manual_labeling_reviews.csv", index=False)

# Combine labeled data with high-confidence pseudo-labeled data
combined_data = pd.concat([labeled_data[['review_text', 'sentiment']], pseudo_labeled_df[['review_text', 'sentiment']]])

# Shuffle dataset before training
combined_data = combined_data.sample(frac=1, random_state=42)

# Save final dataset for training
combined_data.to_csv("data/ssl_training_dataset.csv", index=False)

print("Pseudo-labeling complete! Datasets saved:")
print("- High-confidence pseudo-labeled reviews → pseudo_labeled_reviews.csv")
print("- Low-confidence reviews for manual check → manual_labeling_reviews.csv")
print("- Final SSL training dataset → ssl_training_dataset.csv")
