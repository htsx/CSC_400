import pandas as pd
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import nltk
import os
import re

# Download required NLTK data
nltk.download(['sentiwordnet', 'stopwords', 'averaged_perceptron_tagger', 'punkt'], quiet=True)

# Initialize constants
INTENSIFIERS = {
    "very": 2.0, "extremely": 2.3, "absolutely": 2.7, "highly": 1.3, 
    "really": 1.9, "so much": 2.4, "too much": 2.2, "incredibly": 2.3, 
    "seriously": 1.7, "literally": 1.7, "unbelievably": 2.5, "super": 1.3, 
    "genuinely": 1.4, "remarkably": 1.7, "exceptionally": 2.0, "awfully": 2.0,
    "terribly": 2.0, "insanely": 2.3, "ridiculously": 2.4, "outstandingly": 2.3,
    "particularly": 1.3, "especially": 1.5, "somewhat": 0.6, "slightly": 0.6,"totally": 2.2,
    "completely": 2.4,"utterly": 2.6,"immensely": 2.1,"overly": 1.4,"fairly": 1.1,"pretty": 1.2,
    "decidedly": 1.7,"dramatically": 2.0,"strongly": 1.8,"severely": 2.3,"profoundly": 2.5,"extensively": 2.1
}

NEGATIONS = {"not", "never", "no", "none", "n't", "neither", "nor", "nothing", "nowhere", "hardly", "scarcely", "barely"}
STOPWORDS = set(stopwords.words('english')) - NEGATIONS

def clean_text(text):
    """Clean the input text by removing unwanted characters, URLs, and HTML tags."""
    # Remove URLs from the text
    text = re.sub(r'http\S+|www\S+', '', text)  
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)  
    # Remove any characters that are not alphanumeric or punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)  
    # Remove extra spaces and strip the text of leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()    
    return text.lower()  # Return the cleaned text in lowercase

def get_sentiment_score(word, pos_tag):
    """Get SentiWordNet scores with POS filtering."""
    pos_map = {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}
    pos = pos_map.get(pos_tag[0]) if pos_tag else None
    
    if not pos:
        return 0.0
    
    synsets = list(swn.senti_synsets(word, pos))
    return synsets[0].pos_score() - synsets[0].neg_score() if synsets else 0.0

# Define soft keywords
soft_positive_keywords = ['smooth', 'easy', 'pleasant', 'efficient', 'well organized', 'quick', 'friendly', 'helpful', 'good service']
soft_negative_keywords = ['a bit late', 'minor issue', 'could be better', 'not great', 'slightly delayed', 'room for improvement']
soft_neutral_keywords = ['nothing special', 'just fine', 'it was ok', 'no big deal', 'neither good nor bad', 'as expected', 'typical experience', 'average at best', 'nothing to complain about']


def analyze_sentiment(text):
    """Hybrid SentiWordNet + Rule-Based analyzer with keyword boosts and soft keywords."""
    try:
        tokens = word_tokenize(str(text).lower())
        pos_tags = pos_tag(tokens)
        
        total_score = 0
        negation = False
        intensifier = 3.6

        # Check for multi-word intensifiers
        for phrase, multiplier in INTENSIFIERS.items():
            if phrase in text.lower():
                intensifier = multiplier
                break
        
        for word, tag in pos_tags:
            if word in STOPWORDS and word not in NEGATIONS:
                continue

            if word in NEGATIONS:
                negation = True
                continue

            if word in INTENSIFIERS:
                intensifier = INTENSIFIERS[word]
                continue

            word_score = get_sentiment_score(word, tag)

            if negation:
                word_score *= -1
                negation = False

            if intensifier != 1.0:
                word_score *= intensifier
                intensifier = 1.0

            total_score += word_score

        # Strong keyword boosts
        strong_positive = {"amazing", "excellent", "fantastic", "loved", "perfect", "best", "very nice"}
        strong_negative = {"horrible", "terrible", "worst", "disgusting", "awful"}

        for kw in strong_positive:
            if kw in text.lower():
                total_score += 5.8
        for kw in strong_negative:
            if kw in text.lower():
                total_score -= 5.8

        # Soft keyword sentiment boosts/penalties (full-text)
        for word in soft_positive_keywords:
            if word in tokens:
                total_score += 1.0
        for word in soft_negative_keywords:
            if word in tokens:
                total_score -= 1.0
        for word in soft_neutral_keywords:
            if word in tokens:
                total_score -= 0.6

        # Final sentiment decision
        if total_score > 1.8:
            return "positive"
        elif total_score < -0.7:
            return "negative"
        return "neutral"

    except Exception:
        return "neutral"
    
def process_dataset(input_path, output_path):
    """Process the validation set with correct column naming"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Load the dataset
        df = pd.read_csv(input_path)

        if 'review_text' not in df.columns:
            raise ValueError("Input file must contain a 'review_text' column")

        # Clean the review text
        df['cleaned_review_text'] = df['review_text'].apply(clean_text)

        # Analyze sentiment
        df['overall_sentiment'] = df['cleaned_review_text'].apply(analyze_sentiment)

        # Reorder columns: put review_text, sentiment, and cleaned review first
        output_cols = ['review_text', 'overall_sentiment', 'cleaned_review_text'] + \
                      [col for col in df.columns if col not in ('review_text', 'overall_sentiment', 'cleaned_review_text')]

        # Save to CSV
        df[output_cols].to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    input_file = "../../data/dataset/utest_set.csv"
    output_file = "../../data/rulebased/t-r_results.csv"
    process_dataset(input_file, output_file)

