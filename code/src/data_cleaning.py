import pandas as pd
import re

def clean_text(text):
    #Removes unwanted characters and normalizes text.
    text = re.sub(r'\s+', ' ', text)  #Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  #Remove punctuation
    return text.lower()

def preprocess_data(filepath):
    #Loads and preprocesses the dataset.
    data = pd.read_csv(filepath)
    data['cleaned_text'] = data['text'].apply(clean_text)
    return data

if __name__ == "__main__":
    input_path = '../data/tripadvisor_reviews.csv'
    output_path = '../data/cleaned_reviews.csv'
    cleaned_data = preprocess_data(input_path)
    cleaned_data.to_csv(output_path, index=False)
