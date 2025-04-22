import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd
import os
import re

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to WordNet POS tag"""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

class SentiWordNetAnalyzer:
    def __init__(self):
        # Optimized thresholds based on your evaluation metrics
        self.pos_threshold = 0.12  # Lowered to capture weak positives
        self.neg_threshold = 0.07  # Reduced to identify more negatives
        self.neutral_threshold = 0.02  # Tighter neutral boundary
        self.min_sentiment_words = 2  # Minimum words to trust sentiment
        
        # Negation handling (core WSD technique)
        self.negation_words = {"not", "no", "never", "n't"}
        self.negation_scope = 4  # Words affected after negation

    def analyze_review(self, review_text):
        review_text = clean_text(review_text)
        tokens = word_tokenize(review_text)
        tagged = pos_tag(tokens)

        total_pos_score = 0.0
        total_neg_score = 0.0
        count = 0
        negate_counter = 0

        for i, (word, tag) in enumerate(tagged):
            wn_tag = get_wordnet_pos(tag)
            if not wn_tag:
                continue

            # Negation handling (preserves WSD integrity)
            if word.lower() in self.negation_words:
                negate_counter = self.negation_scope
                continue

            synset = lesk(tokens, word, pos=wn_tag)
            if not synset:
                continue

            try:
                swn_synset = swn.senti_synset(synset.name())
                pos_score = swn_synset.pos_score()
                neg_score = swn_synset.neg_score()

                # Negation flip (standard WSD practice)
                if negate_counter > 0:
                    pos_score, neg_score = neg_score, pos_score
                    negate_counter -= 1

                # Confidence-weighted scoring (WSD-compatible)
                weight = abs(pos_score - neg_score)
                total_pos_score += pos_score * weight
                total_neg_score += neg_score * weight
                count += 1
            except:
                continue

        # WSD-faithful neutral fallback
        if count < self.min_sentiment_words:
            return "neutral", {'positive': 0.0, 'negative': 0.0}

        avg_pos = total_pos_score / count
        avg_neg = total_neg_score / count

        # Threshold-based classification (maintains WSD logic)
        if avg_pos >= self.pos_threshold and (avg_pos - avg_neg) > self.neutral_threshold:
            sentiment = 'positive'
        elif avg_neg >= self.neg_threshold and (avg_neg - avg_pos) > self.neutral_threshold:
            sentiment = 'negative'
        elif abs(avg_pos - avg_neg) <= self.neutral_threshold:
            sentiment = 'neutral'
        else:
            sentiment = 'neutral'

        return sentiment, {'positive': round(avg_pos, 4), 'negative': round(avg_neg, 4)}

if __name__ == "__main__":
    input_file = "../../data/dataset/uvalidation_set.csv"
    output_file = "../../data/wordsense/v-ws_results.csv"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if 'review_text' not in df.columns:
        raise ValueError("Expected column 'review_text' not found in CSV.")

    analyzer = SentiWordNetAnalyzer()

    results = []

    for _, row in df.iterrows():
        review = row['review_text']
        if isinstance(review, str) and review.strip():
            overall_sentiment, sentiment_scores = analyzer.analyze_review(review)
            result = {
                'review_name': row.get('review_name', ''),
                'review_type': row.get('review_type', ''),
                'passenger_name': row.get('passenger_name', ''),
                'review_date': row.get('review_date', ''),
                'review_text': review,
                'overall_emotion': overall_sentiment,
            }
            result.update(sentiment_scores)
            results.append(result)

    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
