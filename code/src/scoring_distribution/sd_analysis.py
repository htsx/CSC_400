import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

#Load the cleaned dataset (make sure it’s ready for analysis)
df = pd.read_csv("../../data/webscrapper/cleaned_skytrax_reviews.csv")

#Initialize VADER Sentiment Analyzer (VADER is pretty good for social media-style text)
vader_analyzer = SentimentIntensityAnalyzer()

#Function to analyze sentiment using VADER
def vader_sentiment(text):
    if pd.isna(text):  #Handle missing values (we don't want errors if there's no text)
        return "Neutral"  #If there's no text, it's considered neutral
    score = vader_analyzer.polarity_scores(str(text))["compound"]  #Get the compound score (overall sentiment)
    if score >= 0.05:  #If the score is positive, return "Positive"
        return "Positive"
    elif score <= -0.05:  #If the score is negative, return "Negative"
        return "Negative"
    else:  #Otherwise, consider it neutral
        return "Neutral"

#Function to analyze sentiment using TextBlob
def textblob_sentiment(text):
    if pd.isna(text):  #Handle missing values (again, no errors if text is missing)
        return "Neutral"  #Default to neutral if there's no text
    score = TextBlob(str(text)).sentiment.polarity  #Get the polarity score from TextBlob
    if score > 0:  #If the score is positive, return "Positive"
        return "Positive"
    elif score < 0:  #If the score is negative, return "Negative"
        return "Negative"
    else:  #Otherwise, consider it neutral
        return "Neutral"

#Apply sentiment analysis to the review text using both VADER and TextBlob
df["VADER_Sentiment"] = df["review_text"].apply(vader_sentiment)  #Using VADER for sentiment
df["TextBlob_Sentiment"] = df["review_text"].apply(textblob_sentiment)  #Using TextBlob for sentiment

#Save the results to a new CSV file for later use
df.to_csv("../../data/scoring_distribution/sd_results.csv", index=False)
print("✅ Sentiment analysis completed. Results saved to data/scoring_distribution/sentiment_results.csv.")  #Let the user know it's done
