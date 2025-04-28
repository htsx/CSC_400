from transformers import pipeline
import pandas as pd
import re
from tqdm import tqdm

# Initialize tqdm for progress tracking
tqdm.pandas()

# Initialize the sentiment analysis pipeline using the pre-trained model
sentiment_checker = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=0  # Set to 0 for GPU, -1 for CPU
)

# Define label mapping
label_mapping = {
    'LABEL_0': 'NEGATIVE',
    'LABEL_1': 'NEUTRAL',
    'LABEL_2': 'POSITIVE'
}

# Define strong and soft keyword sets
strong_positive = {"amazing", "excellent", "fantastic", "loved", "perfect", "Best", "very nice"}
strong_negative = {"horrible", "terrible", "worst", "disgusting", "awful"}
strong_neutral = { "okay", "fine", "decent", "average", "neutral", "ok", "standard", "typical", "ordinary", "moderate", "acceptable", "satisfactory", "clean", "on time", "punctual", "smooth" }

soft_positive_keywords = {
    "great", "smooth", "friendly", "quick", "well organized", "no issues", "pleasant", "lovely", "enjoyable", "good"
}

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text.lower()

def adjust_threshold(sentiment, confidence,
                     positive_threshold=0.96,
                     negative_threshold=0.67,
                     neutral_threshold=0.58):
    if sentiment == 'POSITIVE' and confidence < positive_threshold:
        return 'NEUTRAL', 0.5
    elif sentiment == 'NEGATIVE' and confidence < negative_threshold:
        return 'NEUTRAL', 0.5
    elif sentiment == 'NEUTRAL' and confidence < neutral_threshold:
        return 'NEUTRAL', max(confidence, 0.6)
    return sentiment, confidence

def keyword_override(text, sentiment, confidence, threshold=0.65):
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
    if sentiment == 'NEUTRAL':
        lowered = text.lower()
        if len(text.split()) >= 10 and any(kw in lowered for kw in soft_positive_keywords):
            return 'POSITIVE', max(confidence, 0.87)
    return sentiment, confidence

def process_reviews(df):
    cleaned_texts = df['review_text'].astype(str).progress_apply(clean_text).tolist()
    model_outputs = sentiment_checker(cleaned_texts, batch_size=32)

    sentiments = []
    confidences = []

    for text, output in zip(cleaned_texts, model_outputs):
        sentiment = label_mapping.get(output['label'], 'NEUTRAL')
        confidence = round(output['score'], 2)

        sentiment, confidence = adjust_threshold(sentiment, confidence)
        sentiment, confidence = keyword_override(text, sentiment, confidence)
        sentiment, confidence = keyword_boost(text, sentiment, confidence)

        sentiments.append(sentiment.capitalize())
        confidences.append(confidence)

    df['sentiment'] = sentiments
    df['confidence'] = confidences
    return df

# Load and process the data
reviews_data = pd.read_csv("../../data/dataset/uvalidation_set.csv")
reviews_data = reviews_data[reviews_data['review_text'].notna()]
reviews_data['review_text'] = reviews_data['review_text'].astype(str)

# Run sentiment analysis and save results
results_df = process_reviews(reviews_data)
results_df.to_csv("../../data/deeplearning/v-dl_results.csv", index=False)

print(f"Analysis complete! {len(results_df)} reviews processed.")