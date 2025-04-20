import pandas as pd
from transformers import pipeline, DistilBertTokenizer

#Set up the Hugging Face model and tokenizer for sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#Load the data that already has hybrid sentiment labels
hybrid_labeled_df = pd.read_csv("../data/extra/hybrid_labeled.csv")

#Get reviews that still need labeling (those marked 'ManualCheck')
unlabeled_reviews = hybrid_labeled_df[hybrid_labeled_df['hybrid_sentiment'] == 'ManualCheck']

#Function to check if a review is too long (based on the token limit of 512 tokens)
def is_review_too_long(review, max_length=512):
    #Tokenize and check if the review exceeds the token limit
    tokens = tokenizer.encode(review, add_special_tokens=True)
    return len(tokens) > max_length  #Return True if the review is too long

#Collect indexes of reviews that are too long, so we can remove them later
indexes_to_remove = []

#Start pseudo-labeling the reviews that need it
for idx, row in unlabeled_reviews.iterrows():
    review_text = row['review_text']
    
    #Skip reviews that are too long (more than 512 tokens) and mark them for removal
    if is_review_too_long(review_text):
        print(f"Removing long review at index {idx}: {review_text[:50]}...")  #Print first 50 chars for logging
        indexes_to_remove.append(idx)
        continue  #Skip this review and move to the next one
    
    #Get the sentiment prediction (either 'POSITIVE' or 'NEGATIVE')
    sentiment = sentiment_analyzer(review_text)[0]['label']  #Get sentiment (either POSITIVE or NEGATIVE)
    
    #Find the review in the dataframe and update the sentiment labels
    matching_row = hybrid_labeled_df[hybrid_labeled_df['review_text'] == review_text]
    if not matching_row.empty:
        hybrid_labeled_df.loc[matching_row.index, 'hybrid_sentiment'] = sentiment
        hybrid_labeled_df.loc[matching_row.index, 'review_classification'] = sentiment  #Update the classification column

#Remove the long reviews using the indexes we collected earlier
hybrid_labeled_df = hybrid_labeled_df.drop(indexes_to_remove)

#Now let's see if there are any reviews that are still marked as 'ManualCheck' after processing
manual_check_reviews = hybrid_labeled_df[
    (hybrid_labeled_df['hybrid_sentiment'] == 'ManualCheck') | 
    (hybrid_labeled_df['review_classification'] == 'ManualCheck')
]

if not manual_check_reviews.empty:
    print(f"🔎 Found {len(manual_check_reviews)} reviews still needing manual labeling:")
    print(manual_check_reviews[['review_text', 'hybrid_sentiment', 'review_classification']].head(10))  #Show first 10 reviews
else:
    print("✅ All reviews have been labeled! No manual check required.")

#Save the updated dataset with pseudo-labeling
hybrid_labeled_df.to_csv("../data/extra/ground_truth_reviews.csv", index=False)

print("✅ Pseudo-labeling complete and saved to 'ground_truth_reviews.csv'")