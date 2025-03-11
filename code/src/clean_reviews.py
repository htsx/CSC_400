import pandas as pd
import re

# Load the CSV file
input_file = 'data/raw_skytrax_reviews.csv'
output_file = 'data/cleaned_skytrax_reviews.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Function to clean review text
def clean_review(text):
    if pd.isna(text):
        return text
    # Remove any 'Verified' or 'Not Verified' text and similar patterns ending with '|'
    cleaned_text = re.sub(r'^\s*(✅Trip Verified\||❎Unverified\||❎Not Verified\||Not Verified\||✅Verified Review\|\s*)', '', text)
    return cleaned_text.strip()

# Apply cleaning function to the review column (assuming it's named 'review_text')
df['review_text'] = df['review_text'].apply(clean_review)

# Save cleaned data to a new CSV
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Cleaned reviews saved to {output_file}")