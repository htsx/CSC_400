import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re

# Download the Punkt tokenizer for TextBlob
nltk.download('punkt')

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Function to clean the review text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.strip()
    return ""

# Keyword-based sentiment analysis
def keyword_based_sentiment(text):
    positive_keywords = ['great', 'excellent', 'friendly', 'smooth', 'comfortable', 'helpful', 'clean']
    negative_keywords = ['terrible', 'delay', 'rude', 'dirty', 'horrible', 'bad', 'awful']

    text_lower = text.lower()
    pos_hits = sum(1 for word in positive_keywords if word in text_lower)
    neg_hits = sum(1 for word in negative_keywords if word in text_lower)

    if pos_hits > neg_hits:
        return 'Positive'
    elif neg_hits > pos_hits:
        return 'Negative'
    else:
        return 'Neutral'

# TextBlob sentiment
def textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# VADER sentiment
def vader_sentiment(text):
    score = vader.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# File paths
input_file = "../../data/webscrapper/cleaned_skytrax_reviews.csv"
output_file = "../../data/webscrapper/hybrid_labeled.csv"

try:
    df = pd.read_csv(input_file)

    # Clean the review text
    df['review_text'] = df['review_text'].apply(clean_text)

    # Apply all three techniques
    df['textblob_label'] = df['review_text'].apply(textblob_sentiment)
    df['vader_label'] = df['review_text'].apply(vader_sentiment)
    df['keyword_label'] = df['review_text'].apply(keyword_based_sentiment)

    # Strict hybrid labeling: all three must agree
    def hybrid_label(row):
        labels = [row['textblob_label'], row['vader_label'], row['keyword_label']]
        if labels.count(labels[0]) == 3:
            return labels[0]
        return 'ManualCheck'

    df['hybrid_sentiment'] = df.apply(hybrid_label, axis=1)

    df.drop(columns=['review_classification'], errors='ignore', inplace=True)
    df['review_classification'] = df['hybrid_sentiment']

    df.to_csv(output_file, index=False)
    print(f"✅ Hybrid labeled data saved to: {output_file}")

    manual_check_df = df[df['hybrid_sentiment'] == 'ManualCheck']
    if not manual_check_df.empty:
        manual_check_file = "../../data/webscrapper/review_needed.csv"
        manual_check_df.to_csv(manual_check_file, index=False)
        print(f"⚠️ Rows needing manual review saved to: {manual_check_file}")
    else:
        print("No reviews need manual check.")

except FileNotFoundError:
    print(f"❌ File not found: {input_file}")
except Exception as e:
    print(f"❌ Error: {e}")
