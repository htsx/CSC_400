import pandas as pd
from transformers import pipeline, AutoTokenizer

# Load a fine-tuned RoBERTa model for sentiment analysis
model_name = "textattack/roberta-base-imdb"  # Fine-tuned for sentiment classification
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=tokenizer
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
        
        # Get sentiment prediction from model
        result = sentiment_pipeline(text)[0]
        
        # RoBERTa IMDB model provides binary output (POSITIVE or NEGATIVE)
        sentiment = result["label"].upper()
        score = round(result["score"], 2)

        # Map to three categories by adjusting threshold
        if score < 0.6:
            sentiment = "NEUTRAL"

        # Construct scores dictionary
        all_scores = {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 0.0}
        all_scores[sentiment] = score

        return sentiment, score, all_scores
    except Exception as e:
        print(f"Error analyzing review: {text[:100]}... Error: {e}")
        return None, None, None

# Load cleaned dataset
input_file = "../../data/webscrapper/cleaned_skytrax_reviews.csv"
df = pd.read_csv(input_file)

# Ensure 'review_text' column exists
if 'review_text' not in df.columns:
    raise ValueError("CSV file must contain a 'review_text' column.")

# Limit to the first 1000 reviews
df = df.head(1001)

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
output_file = "../../data/deep_learning/dl_results.csv"
results_df.to_csv
