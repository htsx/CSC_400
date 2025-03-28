import pandas as pd
from transformers import pipeline, AutoTokenizer

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=tokenizer,
    top_k=None  # Fixes deprecation warning
)

# Define neutral keywords
neutral_keywords = ["ok", "meh", "decent"]

# Function to truncate text properly using the tokenizer
def truncate_text(text, max_tokens=512):
    encoded = tokenizer(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

# Function to analyze sentiment
def analyze_sentiment(text):
    try:
        text = truncate_text(text)  # Truncate properly with tokenizer
        
        # Check for neutral keywords
        if any(word in text.lower() for word in neutral_keywords):
            return "NEUTRAL", 0.99, {'NEGATIVE': 0.00, 'NEUTRAL': 0.99, 'POSITIVE': 0.00}
        
        # Get all scores from the model
        result = sentiment_pipeline(text)[0]  # Extract first result list
        label_map = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'}
        
        # Extract and round all scores
        all_scores = {label_map[item['label']]: round(item['score'], 2) for item in result}
        
        # Get the highest scoring sentiment
        sentiment = max(all_scores.items(), key=lambda x: x[1])[0]
        score = round(all_scores[sentiment], 2)  # Round confidence score
        
        # Adjust threshold for neutrality
        if score < 0.6:
            sentiment = 'NEUTRAL'
        return sentiment, score, all_scores
    except Exception as e:
        print(f"Error analyzing review: {text[:100]}... Error: {e}")
        return None, None, None

# Load cleaned dataset
input_file = "data/cleaned_skytrax_reviews.csv"
df = pd.read_csv(input_file)

# Ensure 'review_text' column exists
if 'review_text' not in df.columns:
    raise ValueError("CSV file must contain a 'review_text' column.")

# Process reviews
results = []
for index, review in df.iterrows():
    sentiment, score, all_scores = analyze_sentiment(review['review_text'])
    results.append({
        'review_name': review['review_name'],
        'review_type': review['review_type'],
        'passenger_name': review['passenger_name'],
        'review_date': review['review_date'],
        'review_text': review['review_text'],
        'sentiment': sentiment,
        'confidence': score,
        'POS': all_scores.get('POSITIVE', 0.0),
        'NEU': all_scores.get('NEUTRAL', 0.0),
        'NEG': all_scores.get('NEGATIVE', 0.0)
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the updated DataFrame back to the CSV
output_file = "data/dl_results.csv"
results_df.to_csv(output_file, index=False)

print(f"Sentiment analysis completed. Results saved to {output_file}.")
