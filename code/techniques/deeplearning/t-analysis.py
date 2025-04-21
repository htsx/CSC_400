from transformers import pipeline
import pandas as pd
import re

#Initialize the sentiment analysis pipeline using the pre-trained model for sentiment classification
sentiment_checker = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

#Define a label mapping for sentiment categories: Negative, Neutral, and Positive
label_mapping = {
    'LABEL_0': 'NEGATIVE',  #Label for negative sentiment
    'LABEL_1': 'NEUTRAL',   #Label for neutral sentiment
    'LABEL_2': 'POSITIVE'   #Label for positive sentiment
}

#Define strong sentiment keywords for overriding model predictions based on keywords in the text
strong_positive = {"amazing", "excellent", "fantastic", "loved", "perfect", "Best", "very nice"}
strong_negative = {"horrible", "terrible", "worst", "disgusting", "awful"}
strong_neutral = { "okay", "fine", "decent", "average", "neutral", "ok", "standard", "typical", "ordinary", "moderate", "acceptable", "satisfactory", "clean", "on time", "punctual", "smooth"}

#Define soft boost keywords that help promote a neutral sentiment to positive if they appear in the text
soft_positive_keywords = {
    "great", "smooth", "friendly", "quick", "well organized", "no issues", "pleasant", "lovely", "enjoyable", "good"
}

def clean_text(text):
    """Clean the input text by removing unwanted characters, URLs, and HTML tags."""
    #Remove URLs from the text
    text = re.sub(r'http\S+|www\S+', '', text)  
    #Remove HTML tags
    text = re.sub(r'<.*?>', '', text)  
    #Remove any characters that are not alphanumeric or punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)  
    #Remove extra spaces and strip the text of leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()  
    return text.lower()  #Return the cleaned text in lowercase

def adjust_threshold(sentiment, confidence,
                     positive_threshold=0.99,
                     negative_threshold=0.75,
                     neutral_threshold=0.55):
    """Adjust the sentiment classification based on the confidence score using predefined thresholds."""
    if sentiment == 'POSITIVE' and confidence < positive_threshold:
        return 'NEUTRAL', 0.5  #Adjust to neutral if confidence is below the threshold for positive
    elif sentiment == 'NEGATIVE' and confidence < negative_threshold:
        return 'NEUTRAL', 0.5  #Adjust to neutral if confidence is below the threshold for negative
    elif sentiment == 'NEUTRAL' and confidence < neutral_threshold:
        return 'NEUTRAL', max(confidence, 0.75)  #Adjust neutral confidence to a minimum of 0.75
    return sentiment, confidence  #Return the original sentiment if no adjustment is needed

def keyword_override(text, sentiment, confidence, threshold=0.60):
    """Override sentiment prediction based on strong sentiment keywords when confidence is low."""
    tokens = text.lower().split()  #Split the cleaned text into tokens (words)
    #If confidence is low, check for strong keywords in the text
    if confidence < threshold:
        if any(word in tokens for word in strong_positive):
            return "POSITIVE", 0.75  #Override to positive if strong positive keyword is found
        elif any(word in tokens for word in strong_negative):
            return "NEGATIVE", 0.75  #Override to negative if strong negative keyword is found
        elif any(word in tokens for word in strong_neutral):
            return "NEUTRAL", 0.75  #Override to neutral if strong neutral keyword is found
    return sentiment, confidence  #Return the original sentiment if no override is applied

def keyword_boost(text, sentiment, confidence):
    """Boost sentiment classification for neutral reviews if soft positive keywords are found."""
    if sentiment == 'NEUTRAL':  #If the sentiment is neutral
        lowered = text.lower()  #Convert the text to lowercase
        #If any soft positive keyword is found, boost the sentiment to positive
        if any(kw in lowered for kw in soft_positive_keywords):
            return 'POSITIVE', max(confidence, 0.82)  #Boost confidence for positive sentiment
    return sentiment, confidence  #Return the original sentiment if no boost is applied

def analyze_sentiment(text):
    """Analyze sentiment of the text and return the predicted sentiment and confidence score."""
    cleaned_text = clean_text(text)  #Clean the review text before analysis
    result = sentiment_checker(cleaned_text)[0]  #Use the pre-trained sentiment analysis model
    
    #Map the model's label to our custom sentiment labels
    sentiment = label_mapping.get(result['label'], 'NEUTRAL')
    confidence = round(result['score'], 2)  #Round confidence score for readability

    #Apply thresholds and keyword logic to adjust the sentiment and confidence
    sentiment, confidence = adjust_threshold(sentiment, confidence)
    sentiment, confidence = keyword_override(cleaned_text, sentiment, confidence)
    sentiment, confidence = keyword_boost(cleaned_text, sentiment, confidence)

    return sentiment.capitalize(), confidence  #Capitalize sentiment for readability and return it with the confidence

def process_reviews(df):
    """Process the reviews in the dataframe by analyzing the sentiment of each review."""
    results = []  #Create an empty list to store the results
    for _, row in df.iterrows():
        sentiment, confidence = analyze_sentiment(row['review_text'])  #Analyze the sentiment of each review
        #Append the results of the analysis to the results list
        results.append({
            'review_name': row['review_name'],
            'review_type': row['review_type'],
            'passenger_name': row['passenger_name'],
            'review_date': row['review_date'],
            'review_text': row['review_text'],
            'sentiment': sentiment,
            'confidence': confidence
        })
    return pd.DataFrame(results)  #Return the results as a pandas DataFrame

#Load and process the reviews data from a CSV file
reviews_data = pd.read_csv("../../data/dataset/utest_set.csv")
reviews_data = reviews_data[reviews_data['review_text'].notna()]  #Remove rows with missing review text
reviews_data['review_text'] = reviews_data['review_text'].astype(str)  #Ensure the review text is a string

#Process the reviews and store the results in a new dataframe
results_df = process_reviews(reviews_data)
#Save the results to a CSV file for further analysis or reporting
results_df.to_csv("../../data/deeplearning/t-dl_results.csv", index=False)

print(f"Analysis complete! {len(results_df)} reviews processed.")  #Print the completion message
