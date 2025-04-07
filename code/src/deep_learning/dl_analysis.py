import pandas as pd
from transformers import pipeline

#Load the sentiment analysis model for Twitter-based sentiment classification
sentiment_checker = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

#List of neutral words like "ok", "meh", "decent" that we don’t want to count as positive or negative
neutral_words = ["ok", "meh", "decent"]

def has_neutral_words(text):
    text_lower = text.lower()  #Make everything lowercase so the comparison is case-insensitive
    return any(word in text_lower for word in neutral_words)

def get_sentiment(review_text):
    try:
        if has_neutral_words(review_text):  #If the review has any neutral words, label it as neutral
            return "NEUTRAL", 0.99, {'POSITIVE': 0.0, 'NEUTRAL': 0.99, 'NEGATIVE': 0.0}
        
        #Get sentiment analysis from the model (limit review length to 1000 characters just in case)
        result = sentiment_checker(review_text[:1000])[0]  
        sentiment = result['label'].upper()  #Get the sentiment label and make sure it’s uppercase
        confidence = round(result['score'], 2)  #Round the confidence score to 2 decimal places

        if confidence < 0.55:  #If the model isn’t confident enough, mark it as neutral
            sentiment = "NEUTRAL"

        #Create a dictionary for sentiment scores
        scores = {'POSITIVE': 0.0, 'NEUTRAL': 0.0, 'NEGATIVE': 0.0}
        scores[sentiment] = confidence  #Set the score for the sentiment we got

        return sentiment, confidence, scores
    except Exception as e:
        print(f"Error analyzing review: {e}")  #Print any errors that come up
        return None, None, None

#Load the review data from the CSV file
reviews_data = pd.read_csv("../../data/webscrapper/cleaned_skytrax_reviews.csv")

if 'review_text' not in reviews_data.columns:
    print("Error: The CSV file needs a 'review_text' column.")  #Make sure the data has the right column
    exit()

#Go through each review, analyze it, and store the results
analysis_results = []
for _, row in reviews_data.iterrows():
    sentiment, confidence, scores = get_sentiment(row['review_text'])

    analysis_results.append({
        'review_name': row['review_name'],
        'review_type': row['review_type'],
        'passenger_name': row['passenger_name'],
        'review_date': row['review_date'],
        'review_text': row['review_text'],
        'sentiment': sentiment,
        'confidence': confidence,
        'POS': scores['POSITIVE'] if scores else None,
        'NEU': scores['NEUTRAL'] if scores else None,
        'NEG': scores['NEGATIVE'] if scores else None
    })

#Convert the results into a DataFrame and save it to a CSV file
results_df = pd.DataFrame(analysis_results)
results_df.to_csv("../../data/deep_learning/dl_results.csv", index=False)

print("Analysis complete! Results saved to dl_results.csv")  #Let us know when it’s done
