from transformers import pipeline
import pandas as pd
import re

#Initialize the sentiment analysis pipeline using the pre-trained model for sentiment classification
sentiment_checker = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

#Define a label mapping for sentiment categories: Negative, Neutral, and Positive
label_mapping = {
    'LABEL_0': 'NEGATIVE',
    'LABEL_1': 'NEUTRAL',
    'LABEL_2': 'POSITIVE'
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
    #Clean the input text by removing unwanted characters, URLs, and HTML tags.
    text = re.sub(r'http\S+|www\S+', '', text)  
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text.lower()

def adjust_threshold(sentiment, confidence,
                     positive_threshold=0.99,
                     negative_threshold=0.75,
                     neutral_threshold=0.55):
    #Adjust the sentiment classification based on the confidence score using predefined thresholds.
    if sentiment == 'POSITIVE' and confidence < positive_threshold:
        return 'NEUTRAL', 0.5
    elif sentiment == 'NEGATIVE' and confidence < negative_threshold:
        return 'NEUTRAL', 0.5
    elif sentiment == 'NEUTRAL' and confidence < neutral_threshold:
        return 'NEUTRAL', max(confidence, 0.75)
    return sentiment, confidence

def keyword_override(text, sentiment, confidence, threshold=0.60):
   #Override sentiment prediction based on strong sentiment keywords when confidence is low.
    tokens = text.lower().split()
    if confidence < threshold:
        if any(word in tokens for word in strong_positive):
            return "POSITIVE", 0.75
        elif any(word in tokens for word in strong_negative):
            return "NEGATIVE", 0.75
        elif any(word in tokens for word in strong_neutral):
            return "NEUTRAL", 0.75
    return sentiment, confidence

def keyword_boost(text, sentiment, confidence):
    #Boost sentiment classification for neutral reviews if soft positive keywords are found.
    if sentiment == 'NEUTRAL':
        lowered = text.lower()
        if any(kw in lowered for kw in soft_positive_keywords):
            return 'POSITIVE', max(confidence, 0.82)
    return sentiment, confidence

def analyze_sentiment(text):
    #Analyze sentiment of the text and return the predicted sentiment and confidence score.
    cleaned_text = clean_text(text)
    result = sentiment_checker(cleaned_text)[0]
    
    #Map the model's label to our custom sentiment labels
    sentiment = label_mapping.get(result['label'], 'NEUTRAL')
    confidence = round(result['score'], 2)

    #Apply thresholds and keyword logic to adjust the sentiment and confidence
    sentiment, confidence = adjust_threshold(sentiment, confidence)
    sentiment, confidence = keyword_override(cleaned_text, sentiment, confidence)
    sentiment, confidence = keyword_boost(cleaned_text, sentiment, confidence)

    return sentiment.capitalize(), confidence

def process_reviews(df):
    #Process the reviews in the dataframe by analyzing the sentiment of each review.
    results = []
    for _, row in df.iterrows():
        sentiment, confidence = analyze_sentiment(row['review_text'])
        results.append({
            'review_name': row['review_name'],
            'review_type': row['review_type'],
            'passenger_name': row['passenger_name'],
            'review_date': row['review_date'],
            'review_text': row['review_text'],
            'sentiment': sentiment,
            'confidence': confidence
        })
    return pd.DataFrame(results)

#Load and process the reviews data from a CSV file
reviews_data = pd.read_csv("../../data/dataset/uvalidation_set.csv")
reviews_data = reviews_data[reviews_data['review_text'].notna()]
reviews_data['review_text'] = reviews_data['review_text'].astype(str)

#Process the reviews and store the results in a new dataframe
results_df = process_reviews(reviews_data)
#Save the results to a CSV file for further analysis or reporting
results_df.to_csv("../../data/deeplearning/v-dl_results.csv", index=False)

print(f"Analysis complete! {len(results_df)} reviews processed.")
