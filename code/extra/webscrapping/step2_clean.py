import pandas as pd
import re

#Load the raw Skytrax reviews CSV file
input_file = 'data/raw_skytrax_reviews.csv'
output_file = 'data/cleaned_skytrax_reviews.csv'

#Read the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

#Clean up the review text
def clean_review(text):
    if pd.isna(text):  #If there's no review text, just return it as it is
        return text
    #Remove any 'Verified' or 'Not Verified' text and similar patterns ending with '|'
    cleaned_text = re.sub(r'^\s*(✅Trip Verified\||❎Unverified\||❎Not Verified\||Not Verified\||✅Verified Review\|\s*)', '', text)
    return cleaned_text.strip()  #Remove any extra spaces before or after

#Apply the cleaning function to the 'review_text' column (assuming that's the column name)
df['review_text'] = df['review_text'].apply(clean_review)

#Save the cleaned reviews to a new CSV file
df.to_csv(output_file, index=False, encoding='utf-8')

#Notify that the cleaned reviews have been saved
print(f"Cleaned reviews saved to {output_file}")
