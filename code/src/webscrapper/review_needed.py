import pandas as pd
from transformers import pipeline, DistilBertTokenizer

# Initialize the sentiment analysis pipeline and tokenizer from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load your hybrid-labeled data
hybrid_labeled_df = pd.read_csv("../../data/webscrapper/hybrid_labeled.csv")

# Check if there are any reviews that need manual labeling based on the 'hybrid_sentiment' column
unlabeled_reviews = hybrid_labeled_df[hybrid_labeled_df['hybrid_sentiment'] == 'ManualCheck']

# Function to check if the review exceeds token limit (512 tokens)
def is_review_too_long(review, max_length=512):
    # Tokenize the review and get the number of tokens
    tokens = tokenizer.encode(review, add_special_tokens=True)
    return len(tokens) > max_length

# List to collect indexes of reviews to remove
indexes_to_remove = []

# Pseudo-labeling: Apply the pretrained model to unlabeled reviews
for idx, row in unlabeled_reviews.iterrows():
    review_text = row['review_text']
    
    # Skip reviews that are too long (in terms of tokens) and collect their index for removal
    if is_review_too_long(review_text):
        print(f"Removing long review at index {idx}: {review_text[:50]}...")  # Print the first 50 chars for logging
        indexes_to_remove.append(idx)
        continue  # Skip this review
    
    # Get sentiment prediction for each review
    sentiment = sentiment_analyzer(review_text)[0]['label']  # Get the sentiment label
    
    # Map the result back to the 'hybrid_sentiment' column
    hybrid_labeled_df.loc[hybrid_labeled_df['review_text'] == review_text, 'hybrid_sentiment'] = sentiment

# Remove the long reviews from the DataFrame using the indexes collected
hybrid_labeled_df = hybrid_labeled_df.drop(indexes_to_remove)

# Save the newly labeled reviews (hybrid + pseudo-labeling) without long reviews
hybrid_labeled_df.to_csv("../../data/webscrapper/ground_truth_reviews.csv", index=False)

print("✅ Pseudo-labeling complete and saved to 'ground_truth.csv'")
