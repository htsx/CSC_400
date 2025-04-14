from collections import defaultdict
import re
import pandas as pd
import os

def clean_text(text):
    """Basic cleaning of the text data."""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)  # Keep alphanumeric and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

class ReviewAnalyzer:
    def __init__(self):
        # Your keyword/topic dictionaries remain unchanged...
        self.positive_keywords = {
            'excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic',
            'love', 'perfect', 'best', 'awesome', 'happy', 'satisfied',
            'recommend', 'helpful', 'impressed', 'outstanding', 'friendly',
            'clean', 'efficient', 'quick', 'modern', 'convenient', 'easy',
            'comfortable', 'smooth', 'pleasant', 'nice', 'airy', 'brilliant',
            'exceptional', 'superb', 'terrific', 'delightful', 'enjoyable',
            'satisfactory', 'pleased', 'content', 'grateful', 'appreciate',
            'welcoming', 'professional', 'courteous', 'polite', 'attentive',
            'spacious', 'well-maintained', 'organized', 'streamlined', 'hassle-free',
            'great', 'pleasant', 'comfortable', 'efficient', 'nice', 'smooth', 'friendly',
            'smoothless', 'outstanding', 'extensive', 'beautiful', 'comfortable', 
            'superior', 'pleased', 'friendly', 'welcoming', 'clean', 'efficient',
            'great', 'pleasant', 'nice', 'easy', 'perfect', 'quick'
        }
        self.negative_keywords = {
            'poor', 'bad', 'terrible', 'awful', 'horrible', 'disappointing',
            'waste', 'worst', 'hate', 'unhappy', 'dissatisfied', 'avoid',
            'negative', 'useless', 'broken', 'expensive', 'rude', 'dirty',
            'chaotic', 'slow', 'inefficient', 'crowded', 'delayed', 'miserable',
            'uncomfortable', 'unfriendly', 'expensive', 'filthy', 'dump',
            'frustrating', 'annoying', 'inconvenient', 'disorganized', 'chaotic',
            'overpriced', 'unacceptable', 'subpar', 'inadequate', 'unpleasant',
            'stressful', 'confusing', 'disappointing', 'unreliable', 'inconsistent'
        }
        self.neutral_keywords = {
            'ok', 'okay', 'average', 'decent', 'fine', 'sufficient',
            'acceptable', 'nothing special', 'mediocre', 'typical',
            'standard', 'middle of the road', 'so-so', 'alright', 'neutral'
        }
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither',
            'nowhere', 'hardly', 'scarcely', 'barely', 'doesn\'t', 'isn\'t',
            'wasn\'t', 'shouldn\'t', 'wouldn\'t', 'couldn\'t', 'won\'t',
            'can\'t', 'don\'t'
        }
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.3, 'super': 1.4,
            'incredibly': 1.8, 'absolutely': 1.7, 'totally': 1.6,
            'completely': 1.5, 'utterly': 1.9, 'exceptionally': 1.7
        }
        self.diminishers = {
            'slightly': 0.7, 'somewhat': 0.8, 'a bit': 0.8, 'kind of': 0.8,
            'sort of': 0.8, 'rather': 0.9, 'fairly': 0.9, 'relatively': 0.9
        }
        self.topics = {
            # topics remain unchanged
            'staff': ['staff', 'employee', 'personnel', 'service', 'assistance', 'helper', 
                      'security', 'check-in', 'counter', 'agent', 'officer', 'crew',
                      'attendant', 'representative', 'worker', 'team'],
            'facilities': ['wifi', 'restroom', 'bathroom', 'toilet', 'shop', 'restaurant', 
                           'cafe', 'seating', 'chair', 'terminal', 'lounge', 'duty-free',
                           'gate', 'concourse', 'area', 'zone', 'section', 'space',
                           'store', 'outlet', 'food', 'beverage', 'snack'],
            'cleanliness': ['clean', 'dirty', 'filthy', 'hygiene', 'tidy', 'mess', 
                            'maintenance', 'sanitary', 'sanitation', 'garbage',
                            'trash', 'litter', 'smell', 'odor', 'stain'],
            'efficiency': ['queue', 'line', 'wait', 'delay', 'quick', 'fast', 'slow', 
                           'efficient', 'process', 'security check', 'boarding',
                           'disembarking', 'transfer', 'connection', 'time',
                           'speed', 'pace', 'flow'],
            'transport': ['parking', 'bus', 'taxi', 'transport', 'connection', 'transfer',
                          'accessibility', 'shuttle', 'train', 'metro', 'subway',
                          'car', 'vehicle', 'transit', 'commute'],
            'comfort': ['comfortable', 'space', 'crowded', 'quiet', 'noisy', 
                        'temperature', 'air conditioning', 'seating', 'chair',
                        'bench', 'rest', 'relax', 'sleep', 'nap', 'restroom',
                        'bathroom', 'toilet'],
            'staff_checkin': ['check-in', 'counter', 'agent', 'representative', 'staff', 'reception', 
                              'employee', 'service', 'welcome', 'assist', 'attendant', 'customer', 
                              'queue', 'line', 'waiting', 'help', 'support', 'clerk', 'desk', 'greeting'],
            'staff_security': ['security', 'officer', 'agent', 'guard', 'screening', 'checkpoint', 
                               'inspection', 'patrol', 'boarding', 'bag check', 'security check', 
                               'x-ray', 'metal detector', 'passport control', 'surveillance', 'safety', 
                               'crowd control', 'escort', 'control'],
            'facilities_wifi': ['wifi', 'internet', 'connection', 'signal', 'speed', 'network', 'access', 
                                'hotspot', 'router', 'bandwidth', 'reception', 'reliable', 'secure', 
                                'login', 'free wifi', 'connection issues', 'slow internet', 'speed test', 
                                'cafe', 'lounge', 'waiting area'],
        }

    def analyze_review(self, review_text):
        review_text = clean_text(review_text)
        sentiment_score = self._calculate_sentiment_score(review_text)
        topic_sentiments = defaultdict(int)
        topic_mentions = defaultdict(int)

        for topic, topic_words in self.topics.items():
            for word in topic_words:
                if word in review_text:
                    topic_mentions[topic] += 1
                    context = self._get_context(review_text, word)
                    sentiment = self._calculate_context_sentiment(context)
                    if sentiment == 0 and not any(w in context for w in (
                        self.positive_keywords | self.negative_keywords | self.neutral_keywords)):
                        sentiment = sentiment_score
                    topic_sentiments[topic] += sentiment

        if sentiment_score >= 1.1:
            overall_sentiment = "positive"
        elif sentiment_score <= -0.9:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': sentiment_score,
            'topic_sentiments': dict(topic_sentiments),
            'topic_mentions': dict(topic_mentions)
        }

    def _calculate_sentiment_score(self, text):
        words = text.split()
        score = 0
        for i, word in enumerate(words):
            if word in self.negation_words:
                for j in range(i+1, min(i+4, len(words))):
                    if words[j] in self.positive_keywords:
                        score -= 1
                    elif words[j] in self.negative_keywords:
                        score += 1
                continue
            if word in self.intensifiers:
                for j in range(i+1, min(i+3, len(words))):
                    if words[j] in self.positive_keywords:
                        score += self.intensifiers[word]
                    elif words[j] in self.negative_keywords:
                        score -= self.intensifiers[word]
                continue
            if word in self.diminishers:
                for j in range(i+1, min(i+3, len(words))):
                    if words[j] in self.positive_keywords:
                        score += self.diminishers[word]
                    elif words[j] in self.negative_keywords:
                        score -= self.diminishers[word]
                continue
            if word in self.positive_keywords:
                score += 1
            elif word in self.negative_keywords:
                score -= 1
        return score

    def _calculate_context_sentiment(self, context_words):
        score = 0
        for word in context_words:
            if word in self.positive_keywords:
                score += 1
            elif word in self.negative_keywords:
                score -= 1
            elif word in self.neutral_keywords:
                score += 0
        return score

    def _get_context(self, text, word, window_size=7):
        words = text.split()
        try:
            word_index = words.index(word)
            start = max(0, word_index - window_size)
            end = min(len(words), word_index + window_size + 1)
            return set(words[start:end])
        except ValueError:
            return set()


# === CSV Analysis Block Starts Here ===

if __name__ == "__main__":
    input_file = "../../data/webscrapper/unlabeled_reviews.csv"
    output_file = "../../data/keywords_topics/tweaked/kt_results.csv"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if 'review_text' not in df.columns:
        raise ValueError("Expected column 'review_text' not found in CSV.")

    analyzer = ReviewAnalyzer()
    results = []

    for _, row in df.iterrows():
        review = row['review_text']
        if isinstance(review, str) and review.strip():
            result = analyzer.analyze_review(review)
            results.append({
                'review_name': row['review_name'],
                'review_type': row['review_type'],
                'passenger_name': row['passenger_name'],
                'review_date': row['review_date'],
                'review_text': review,
                'overall_sentiment': result['overall_sentiment'],
                'sentiment_score': result['sentiment_score'],
                **result['topic_mentions'],
                **{f"{k}_sentiment": v for k, v in result['topic_sentiments'].items()}
            })

    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
