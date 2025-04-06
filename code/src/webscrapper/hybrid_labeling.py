import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re
nltk.download('punkt')

# Initialize VADER
vader = SentimentIntensityAnalyzer()

<<<<<<< HEAD
<<<<<<< HEAD
# Clean review text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.strip()
    return ""

# Keyword-based sentiment
=======
# Function to clean special characters and NaN values
=======
# Clean review text
>>>>>>> 17bd358c (dataset)
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.strip()
    return ""

<<<<<<< HEAD
# Simple keyword-based function (replace with topic modeling if needed)
>>>>>>> a4d1eadb (commit)
=======
# Keyword-based sentiment
>>>>>>> 17bd358c (dataset)
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

<<<<<<< HEAD
<<<<<<< HEAD
# File paths
=======
# Load your cleaned dataset
>>>>>>> a4d1eadb (commit)
=======
# File paths
>>>>>>> 17bd358c (dataset)
input_file = "../../data/webscrapper/cleaned_skytrax_reviews.csv"
output_file = "../../data/webscrapper/hybrid_labeled.csv"

try:
    df = pd.read_csv(input_file)

<<<<<<< HEAD
<<<<<<< HEAD
    # Clean text
    df['review_text'] = df['review_text'].apply(clean_text)

    # Apply individual models
=======
    # Step 1: Clean the review text (remove NaNs and special characters)
    df['review_text'] = df['review_text'].apply(clean_text)

    # Step 2: Apply all three sentiment techniques
>>>>>>> a4d1eadb (commit)
=======
    # Clean text
    df['review_text'] = df['review_text'].apply(clean_text)

    # Apply individual models
>>>>>>> 17bd358c (dataset)
    df['textblob_label'] = df['review_text'].apply(textblob_sentiment)
    df['vader_label'] = df['review_text'].apply(vader_sentiment)
    df['keyword_label'] = df['review_text'].apply(keyword_based_sentiment)

<<<<<<< HEAD
<<<<<<< HEAD
    # Apply hybrid label logic
    def hybrid_label(row):
        labels = [row['textblob_label'], row['vader_label'], row['keyword_label']]
        if labels.count(labels[0]) == len(labels):
            return labels[0]
=======
    # Final hybrid label
=======
    # Apply hybrid label logic
>>>>>>> 17bd358c (dataset)
    def hybrid_label(row):
        labels = [row['textblob_label'], row['vader_label'], row['keyword_label']]
        if labels.count(labels[0]) == len(labels):
            return labels[0]
<<<<<<< HEAD
        
        # If two models agree, use that label
>>>>>>> a4d1eadb (commit)
=======
>>>>>>> 17bd358c (dataset)
        if labels.count('Positive') >= 2:
            return 'Positive'
        elif labels.count('Negative') >= 2:
            return 'Negative'
        elif labels.count('Neutral') >= 2:
            return 'Neutral'
<<<<<<< HEAD
<<<<<<< HEAD
=======
        
        # If models disagree, flag for manual review
>>>>>>> a4d1eadb (commit)
=======
>>>>>>> 17bd358c (dataset)
        return 'ManualCheck'

    # Apply the hybrid sentiment labeling function
    df['hybrid_sentiment'] = df.apply(hybrid_label, axis=1)

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 17bd358c (dataset)
    # ✅ Ensure review_classification always mirrors hybrid_sentiment
    df.drop(columns=['review_classification'], errors='ignore', inplace=True)
    df['review_classification'] = df['hybrid_sentiment']

    # Save the full labeled data
<<<<<<< HEAD
    df.to_csv(output_file, index=False)
    print(f"✅ Hybrid labeled data saved to: {output_file}")

    # Save reviews that need manual check to a separate file
=======
    # Save the result to the output file
    df.to_csv(output_file, index=False)
    print(f"✅ Hybrid labeled data saved to: {output_file}")

    # Separate the rows needing manual check
>>>>>>> a4d1eadb (commit)
=======
    df.to_csv(output_file, index=False)
    print(f"✅ Hybrid labeled data saved to: {output_file}")

    # Save reviews that need manual check to a separate file
>>>>>>> 17bd358c (dataset)
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
