import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re

#Download the Punkt tokenizer for TextBlob (needed for text processing)
nltk.download('punkt')

#Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

#Function to clean the review text
def clean_text(text):
    if isinstance(text, str):  #Check if the text is a string
        #Remove any non-alphabet characters (just keep letters and spaces)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.strip()  #Strip out any leading/trailing spaces
    return ""  #If it's not a string, return an empty string

#Function for sentiment analysis based on specific keywords
def keyword_based_sentiment(text):
    #List of positive and negative keywords
    positive_keywords = ['great', 'excellent', 'friendly', 'smooth', 'comfortable', 'helpful', 'clean']
    negative_keywords = ['terrible', 'delay', 'rude', 'dirty', 'horrible', 'bad', 'awful']
    
    text_lower = text.lower()  #Make the text lowercase for easier matching
    pos_hits = sum(1 for word in positive_keywords if word in text_lower)  #Count positive keywords
    neg_hits = sum(1 for word in negative_keywords if word in text_lower)  #Count negative keywords

    #Decide sentiment based on which type of keywords occur more
    if pos_hits > neg_hits:
        return 'Positive'
    elif neg_hits > pos_hits:
        return 'Negative'
    else:
        return 'Neutral'

#TextBlob-based sentiment analysis
def textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity  #Get polarity score (-1 to 1)
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

#VADER-based sentiment analysis
def vader_sentiment(text):
    score = vader.polarity_scores(text)['compound']  #Get the compound score
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

#File paths for input and output
input_file = "../../data/webscrapper/cleaned_skytrax_reviews.csv"
output_file = "../../data/webscrapper/hybrid_labeled.csv"

try:
    #Read in the reviews data
    df = pd.read_csv(input_file)

    #Clean the review text column
    df['review_text'] = df['review_text'].apply(clean_text)

    #Apply sentiment models: TextBlob, VADER, and keyword-based
    df['textblob_label'] = df['review_text'].apply(textblob_sentiment)
    df['vader_label'] = df['review_text'].apply(vader_sentiment)
    df['keyword_label'] = df['review_text'].apply(keyword_based_sentiment)

    #Function to apply hybrid labeling based on majority vote
    def hybrid_label(row):
        labels = [row['textblob_label'], row['vader_label'], row['keyword_label']]  #Get the labels from all models
        if labels.count(labels[0]) == len(labels):  #If all labels are the same, use that label
            return labels[0]
        if labels.count('Positive') >= 2:
            return 'Positive'
        elif labels.count('Negative') >= 2:
            return 'Negative'
        elif labels.count('Neutral') >= 2:
            return 'Neutral'
        return 'ManualCheck'  #If no majority, flag for manual check

    #Apply hybrid sentiment label
    df['hybrid_sentiment'] = df.apply(hybrid_label, axis=1)

    #Make sure 'review_classification' column matches the hybrid sentiment
    df.drop(columns=['review_classification'], errors='ignore', inplace=True)
    df['review_classification'] = df['hybrid_sentiment']

    #Save the labeled dataset
    df.to_csv(output_file, index=False)
    print(f"✅ Hybrid labeled data saved to: {output_file}")

    #Save reviews that need a manual check to a separate file
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