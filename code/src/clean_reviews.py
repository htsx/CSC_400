import pandas as pd
import re

#loads the csv file
input_file = 'data/raw_skytrax_reviews.csv'
output_file = 'data/cleaned_skytrax_reviews.csv'

#reads the csv file
df = pd.read_csv(input_file)

#clean review text
def clean_review(text):
    if pd.isna(text):
        return text
    #remove any 'Verified' or 'Not Verified' text and similar patterns ending with '|'
    cleaned_text = re.sub(r'^\s*(✅Trip Verified\||❎Unverified\||❎Not Verified\||Not Verified\||✅Verified Review\|\s*)', '', text)
    return cleaned_text.strip()

#apply cleaning function to the review column (assuming it's named 'review_text')
df['review_text'] = df['review_text'].apply(clean_review)

#save cleaned data to a new csv in data folder
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Cleaned reviews saved to {output_file}")