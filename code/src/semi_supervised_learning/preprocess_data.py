import os
import pandas as pd
from transformers import BertTokenizer

# Define the file paths
input_file = "C:/Users/twoce/OneDrive/CSC 400/csc_400/code/data/ground_truth/ground_truth_reviews.csv"
output_file = "C:/Users/twoce/OneDrive/CSC 400/csc_400/code/src/semi_supervised_learning/data/processed_reviews.csv"

# Load your full dataset
data = pd.read_csv(input_file)

# Check the column names to ensure the expected 'review_text' and 'ground_truth_sentiment' columns exist
print("Columns in CSV:", data.columns)

# Ensure the column names are correct
if 'review_text' not in data.columns or 'ground_truth_sentiment' not in data.columns:
    raise KeyError("The columns 'review_text' or 'ground_truth_sentiment' are missing. Please check your CSV.")

# Initialize the tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize only the first 1000 labeled reviews
def preprocess_text(review_text):
    encoding = tokenizer(review_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    return encoding['input_ids'].tolist()  # Convert tensors to list format

# Initialize 'tokenized' column
data['tokenized'] = None  # Create the 'tokenized' column first

# Only process the first 1000 rows (labeled data)
data.iloc[:1000, data.columns.get_loc('tokenized')] = data['review_text'].iloc[:1000].apply(preprocess_text)

# Save processed data in the same folder as the script (only tokenized labeled data will be saved for now)
data.to_csv(output_file, index=False)

print("Preprocessing complete! Processed data saved to:", output_file)
