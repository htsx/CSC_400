import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def clean_airport_ratings(filepath):
    """Cleans the raw airport ratings data."""
    data = pd.read_csv(filepath)
    
    # Inspect column names
    print(data.columns)
    
    # Ensure there are no NaN or empty values in 'Rating' (as this contains the review)
    data = data.dropna(subset=['Rating'])
    
    # Clean the 'Rating' column (assuming it's textual and requires stripping spaces)
    data['cleaned_text'] = data['Rating'].apply(lambda x: str(x).strip())  # Adjust column name as needed
    
    # Save the cleaned data to a new CSV
    cleaned_filepath = 'data/cleaned_reviews.csv'
    data.to_csv(cleaned_filepath, index=False)
    return cleaned_filepath

def analyze_sentiment(text):
    """Performs sentiment analysis on a single text."""
    if not text or pd.isna(text):
        return 0  # Return neutral sentiment if text is empty or missing
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def add_sentiment_scores(filepath):
    """Adds sentiment scores to the dataset."""
    data = pd.read_csv(filepath)
    
    # Ensure there are no NaN or empty values in 'cleaned_text'
    if 'cleaned_text' not in data.columns:
        raise ValueError("The dataset must contain a 'cleaned_text' column.")

    data['sentiment_score'] = data['cleaned_text'].apply(analyze_sentiment)
    return data

if __name__ == "__main__":
    input_path = 'data/cleaned_reviews.csv'  # Path to input cleaned reviews
    output_path = 'data/sentiment_reviews.csv'  # Path to save sentiment results
    
    # Clean the data from the scraped CSV
    cleaned_data_path = clean_airport_ratings('data/airport_ratings.csv')
    
    # Perform sentiment analysis and save results
    sentiment_data = add_sentiment_scores(cleaned_data_path)
    sentiment_data.to_csv(output_path, index=False)
    print(f"Sentiment scores saved to {output_path}")
