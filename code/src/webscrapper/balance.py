import pandas as pd
from transformers import AutoTokenizer

# Load the dataset
df = pd.read_csv('with_more_neutral.csv')  # Replace with your dataset path

# Initialize the tokenizer for your sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Clean the dataset by addressing empty or invalid fields
# Replace blank or missing passenger names with "Anonymous"
df['passenger_name'] = df['passenger_name'].fillna('Anonymous')

# Remove rows with empty or invalid review text
df = df[df['review_text'].notna() & (df['review_text'].str.strip() != '')]

# Check and correct the review classification, if necessary
df['review_classification'] = df['review_classification'].replace('NotClassified', 'Neutral')

# Check if there are any rows with a completely empty or invalid row structure and drop them
df = df.dropna(subset=['review_classification', 'review_text'])

# Check for any additional invalid or unexpected values in the columns and clean them
df['review_classification'] = df['review_classification'].str.strip()

# Function to check if the review's token length is under 512 tokens
def is_under_token_limit(review_text):
    tokens = tokenizer.encode(review_text)
    return len(tokens) <= 512

# Filter out reviews longer than 512 tokens
df['under_token_limit'] = df['review_text'].apply(is_under_token_limit)
df = df[df['under_token_limit'] == True]

# Filter out each classification to have a maximum of 4000 reviews
positive_reviews = df[df['review_classification'] == 'Positive'].sample(n=4000, random_state=42)
neutral_reviews = df[df['review_classification'] == 'Neutral'].sample(n=4000, random_state=42)
negative_reviews = df[df['review_classification'] == 'Negative'].sample(n=4000, random_state=42)

# Concatenate the three subsets
balanced_df = pd.concat([positive_reviews, neutral_reviews, negative_reviews])

# Save the balanced dataset
balanced_df.to_csv('../../data/webscrapper/balanced.csv', index=False)

print("The dataset has been cleaned and balanced with 4000 reviews for each classification, excluding reviews longer than 512 tokens.")
