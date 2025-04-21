import pandas as pd
from transformers import AutoTokenizer

# Load the original dataset and the generated neutral reviews
main_df = pd.read_csv('data/ground_truth_reviews.csv')  # Replace with your original dataset path
generated_neutral = pd.read_csv('data/openai_generated_neutral_reviews.csv')  # Your synthetic reviews

# Combine them into one DataFrame
df = pd.concat([main_df, generated_neutral], ignore_index=True)

# Tokenizer for distilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Clean the dataset by addressing empty or invalid fields
df['passenger_name'] = df['passenger_name'].fillna('Anonymous')
df = df[df['review_text'].notna() & (df['review_text'].str.strip() != '')]
df['review_classification'] = df['review_classification'].replace('NotClassified', 'Neutral')
df = df.dropna(subset=['review_classification', 'review_text'])
df['review_classification'] = df['review_classification'].str.strip()

# Only keep rows where sentiment labels agree 4 out of 4 (TextBlob, Vader, Keyword, Hybrid)
df_4out4 = df[
    (df['textblob_label'] == df['vader_label']) &
    (df['vader_label'] == df['keyword_label']) &
    (df['keyword_label'] == df['hybrid_sentiment'])
]

# Only keep rows where sentiment labels do not agree completely (3 out of 4)
df_other = df[
    (df['review_classification'].isin(['Positive', 'Negative'])) &
    ((df['textblob_label'] != df['vader_label']) |
    (df['vader_label'] != df['keyword_label']) |
    (df['keyword_label'] != df['hybrid_sentiment']))
]

# Function to check if the review's token length is under 512 tokens
def is_under_token_limit(review_text):
    tokens = tokenizer.encode(review_text)
    return len(tokens) <= 512

# Apply token length check
df_4out4['under_token_limit'] = df_4out4['review_text'].apply(is_under_token_limit)
df_4out4 = df_4out4[df_4out4['under_token_limit'] == True]

df_other['under_token_limit'] = df_other['review_text'].apply(is_under_token_limit)
df_other = df_other[df_other['under_token_limit'] == True]

# Check the available counts for positive and negative reviews in 4-out-4 agreement
positive_reviews_4out4_count = df_4out4[df_4out4['review_classification'] == 'Positive'].shape[0]
negative_reviews_4out4_count = df_4out4[df_4out4['review_classification'] == 'Negative'].shape[0]

# Sample only if there are enough reviews; otherwise, take all available reviews
positive_sample_size_4out4 = min(positive_reviews_4out4_count, 4000)
negative_sample_size_4out4 = min(negative_reviews_4out4_count, 4000)

positive_reviews_4out4 = df_4out4[df_4out4['review_classification'] == 'Positive'].sample(n=positive_sample_size_4out4, random_state=42)
negative_reviews_4out4 = df_4out4[df_4out4['review_classification'] == 'Negative'].sample(n=negative_sample_size_4out4, random_state=42)

# If not enough 4-out-4 reviews for Positive, take additional from review_classification column
if positive_reviews_4out4.shape[0] < 4000:
    remaining_positive_needed = 4000 - positive_reviews_4out4.shape[0]
    additional_positive_reviews = df_other[df_other['review_classification'] == 'Positive'].sample(n=remaining_positive_needed, random_state=42)
    positive_reviews_4out4 = pd.concat([positive_reviews_4out4, additional_positive_reviews])

# If not enough 4-out-4 reviews for Negative, take additional from review_classification column
if negative_reviews_4out4.shape[0] < 4000:
    remaining_negative_needed = 4000 - negative_reviews_4out4.shape[0]
    additional_negative_reviews = df_other[df_other['review_classification'] == 'Negative'].sample(n=remaining_negative_needed, random_state=42)
    negative_reviews_4out4 = pd.concat([negative_reviews_4out4, additional_negative_reviews])

# Collect all neutral reviews (from the generated neutral reviews file)
neutral_reviews = generated_neutral.copy()
neutral_reviews['under_token_limit'] = neutral_reviews['review_text'].apply(is_under_token_limit)
neutral_reviews = neutral_reviews[neutral_reviews['under_token_limit'] == True]

# Check how many neutral reviews we already have in the original dataset
existing_neutral_reviews = df_4out4[df_4out4['review_classification'] == 'Neutral']
existing_neutral_reviews = existing_neutral_reviews[existing_neutral_reviews['under_token_limit'] == True]

# Calculate how many more neutral reviews are needed to reach 4000
remaining_neutral_needed = 4000 - existing_neutral_reviews.shape[0]

# If there are not enough existing neutral reviews, sample from the generated reviews
if remaining_neutral_needed > 0:
    # Ensure you only take as many as needed
    neutral_reviews = neutral_reviews.sample(n=remaining_neutral_needed, random_state=42)

# Combine the existing neutral reviews with the sampled generated neutral reviews
neutral_reviews = pd.concat([existing_neutral_reviews, neutral_reviews])

# Now neutral_reviews will contain exactly 4000 neutral reviews (existing + generated)
neutral_reviews = neutral_reviews.sample(n=4000, random_state=42)  # Ensures randomness

# Combine all reviews into a balanced dataset
balanced_df = pd.concat([positive_reviews_4out4, negative_reviews_4out4, neutral_reviews])

# Save the cleaned, balanced dataset to validation set CSV
balanced_df.to_csv('../../data/dataset/validation_set.csv', index=False)

# Create the unused_reviews DataFrame by removing validation set reviews from the original dataset
unused_reviews = df[~df['review_text'].isin(balanced_df['review_text'])]

# Save the unused reviews to a new CSV file
unused_reviews.to_csv('../testset/data/unused_reviews.csv', index=False)

# Print final details
print(f"Dataset cleaned and balanced with {positive_reviews_4out4.shape[0]} positive reviews, "
      f"{negative_reviews_4out4.shape[0]} negative reviews, and 4000 neutral reviews.")
print(f"Unused reviews saved to 'unused_reviews.csv' with total {unused_reviews.shape[0]} reviews.")
