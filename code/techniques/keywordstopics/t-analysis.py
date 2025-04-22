import nltk
from nltk.corpus import sentiwordnet as swn
from collections import defaultdict
import re
import pandas as pd
import os

# Ensure SentiWordNet is downloaded
nltk.download('sentiwordnet')
nltk.download('wordnet')

def clean_text(text):
    """Basic cleaning of the text data."""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)  # Keep alphanumeric and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

class EmotionBasedReviewAnalyzer:
    def __init__(self):
        # Emotion categories based on words (this can be extended)
        self.emotion_categories = {
            'happiness': ['happy', 'joy', 'excited', 'love', 'pleasure', 'cheerful', 'grateful', 'elated'],
            'sadness': ['sad', 'depressed', 'unhappy', 'down', 'gloomy', 'mournful', 'sorrow', 'disappointed'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'enraged', 'annoyed', 'hostile', 'frustrated'],
            'fear': ['afraid', 'scared', 'fearful', 'nervous', 'anxious', 'worried', 'terrified', 'horrified'],
            'disgust': ['disgusted', 'repelled', 'sickened', 'gross', 'nauseous', 'revolted', 'detestable'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'dumbfounded']
        }

    def analyze_review(self, review_text):
        review_text = clean_text(review_text)
        emotion_scores = defaultdict(float)
        word_count = 0

        # Tokenize the review text into words
        words = review_text.split()

        for word in words:
            # Get the synsets for the word
            synsets = nltk.corpus.wordnet.synsets(word)
            for syn in synsets:
                # Get the SentiWordNet score for the word
                senti_synset = swn.senti_synset(syn.name())
                pos_score = senti_synset.pos_score()
                neg_score = senti_synset.neg_score()

                # Use the SentiWordNet scores to calculate sentiment
                sentiment_score = pos_score - neg_score

                # Check which emotion category the word belongs to
                for emotion, emotion_words in self.emotion_categories.items():
                    if word in emotion_words:
                        emotion_scores[emotion] += sentiment_score
                        word_count += 1

        # Normalize sentiment scores by word count
        if word_count > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= word_count

        # Determine the dominant emotion
        if emotion_scores:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        else:
            dominant_emotion = 'neutral'

        return dominant_emotion, emotion_scores

if __name__ == "__main__":
    input_file = "../../data/dataset/utest_set.csv"
    output_file = "../../data/keywordstopics/t_kt_results_emotion.csv"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if 'review_text' not in df.columns:
        raise ValueError("Expected column 'review_text' not found in CSV.")

    analyzer = EmotionBasedReviewAnalyzer()
    results = []

    for _, row in df.iterrows():
        review = row['review_text']
        if isinstance(review, str) and review.strip():
            dominant_emotion, emotion_scores = analyzer.analyze_review(review)
            result = {
                'review_name': row['review_name'],
                'review_type': row['review_type'],
                'passenger_name': row['passenger_name'],
                'review_date': row['review_date'],
                'review_text': review,
                'dominant_emotion': dominant_emotion,
            }
            # Add emotion scores to the result
            result.update(emotion_scores)
            results.append(result)

    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
