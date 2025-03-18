import pandas as pd
from transformers import pipeline
from tqdm import tqdm  # For progress tracking

# Load the dataset
reviews_df = pd.read_csv('data/cleaned_skytrax_reviews.csv').head(20)  # Limit to 20 rows for testing

# Load a pre-trained sentiment analysis model
# This model is BERT-based and fine-tuned for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Perform sentiment analysis on each review
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text)[0]
        return result['label'], result['score']
    except Exception as e:
        print(f"Error analyzing sentiment for text: {text}. Error: {e}")
        return None, None

# Apply sentiment analysis to the dataset with progress tracking
tqdm.pandas()  # Enable progress_apply for pandas
reviews_df[['deep_learning_sentiment', 'deep_learning_score']] = reviews_df['review_text'].progress_apply(
    lambda text: pd.Series(analyze_sentiment(str(text)))
)

# Save the results to a new CSV file
reviews_df.to_csv('data/dl_results.csv', index=False)

print("Deep learning sentiment analysis complete. Data saved to 'data/dl_results.csv'.")