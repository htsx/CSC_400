import pandas as pd
import re

#Load the cleaned reviews CSV file
input_file = 'data/ground_truth_reviews.csv'
output_file = input_file

#Read the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

#Clean review text
def clean_final_review(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['review_text'] = df['review_text'].apply(clean_final_review)

#Normalize and clean sentiment labels
def normalize_label(label):
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in ['positive', 'pos']:
        return 'Positive'
    elif label in ['negative', 'neg']:
        return 'Negative'
    elif label in ['neutral']:
        return 'Neutral'
    else:
        return None  #Anything unexpected gets dropped later

df['review_classification'] = df['review_classification'].apply(normalize_label)

#Drop rows where label couldn't be normalized
df = df.dropna(subset=['review_classification'])

#Save the cleaned file
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Final cleaned reviews saved to {output_file}")
