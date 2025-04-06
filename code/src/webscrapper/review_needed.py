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
    
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 17bd358c (dataset)
    # Map the result back to the 'hybrid_sentiment' and 'review_classification' columns
    matching_row = hybrid_labeled_df[hybrid_labeled_df['review_text'] == review_text]
    if not matching_row.empty:
        hybrid_labeled_df.loc[matching_row.index, 'hybrid_sentiment'] = sentiment
        hybrid_labeled_df.loc[matching_row.index, 'review_classification'] = sentiment  # Update review_classification
<<<<<<< HEAD
=======
    # Map the result back to the 'hybrid_sentiment' column
    hybrid_labeled_df.loc[hybrid_labeled_df['review_text'] == review_text, 'hybrid_sentiment'] = sentiment
>>>>>>> a4d1eadb (commit)
=======
>>>>>>> 17bd358c (dataset)

# Remove the long reviews from the DataFrame using the indexes collected
hybrid_labeled_df = hybrid_labeled_df.drop(indexes_to_remove)

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 17bd358c (dataset)
# **Print out all rows where either hybrid_sentiment or review_classification is still 'ManualCheck'**
manual_check_reviews = hybrid_labeled_df[
    (hybrid_labeled_df['hybrid_sentiment'] == 'ManualCheck') | 
    (hybrid_labeled_df['review_classification'] == 'ManualCheck')
]

if not manual_check_reviews.empty:
    print(f"🔎 Found {len(manual_check_reviews)} reviews still marked for 'ManualCheck' in either 'hybrid_sentiment' or 'review_classification':")
    print(manual_check_reviews[['review_text', 'hybrid_sentiment', 'review_classification']].head(10))  # Display first 10 reviews
else:
    print("✅ No reviews left needing manual check in either column.")

<<<<<<< HEAD
# Save the newly labeled reviews (hybrid + pseudo-labeling) without long reviews
hybrid_labeled_df.to_csv("../../data/webscrapper/ground_truth_reviews.csv", index=False)

print("✅ Pseudo-labeling complete and saved to 'ground_truth_reviews.csv'")
=======
# Save the newly labeled reviews (hybrid + pseudo-labeling) without long reviews
hybrid_labeled_df.to_csv("../../data/webscrapper/ground_truth_reviews.csv", index=False)

print("✅ Pseudo-labeling complete and saved to 'ground_truth.csv'")
>>>>>>> a4d1eadb (commit)
=======
# Save the newly labeled reviews (hybrid + pseudo-labeling) without long reviews
hybrid_labeled_df.to_csv("../../data/webscrapper/ground_truth_reviews.csv", index=False)

print("✅ Pseudo-labeling complete and saved to 'ground_truth_reviews.csv'")
>>>>>>> 17bd358c (dataset)
