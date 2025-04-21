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
        self.positive_keywords = {
        'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
        'perfect', 'best', 'awesome', 'impressed', 'outstanding',
        'exceptional', 'superb', 'unforgettable', 'splendid',
        'first-class', 'ideal', 'breathtaking', 'flawless'
    }

        self.negative_keywords = {
            'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disgusting', 'unacceptable', 'atrocious', 'horrendous',
            'disgraceful', 'repulsive', 'revolting', 'unbearable',
            'intolerable', 'disastrous', 'appalling', 'detestable',     "delayed", "cancelled", "rude", "lost baggage", "missed connection",
    "unhelpful", "incompetent", "dirty", "cramped", "overbooked",
    "long wait", "no service", "rushed", "chaotic", "lack of communication"
        }

        self.neutral_keywords = {
            "adequate", "sufficient", "standard", "ordinary",
            "unremarkable", "passable", "typical", "expected",
            "average", "fine", "okay", "decent", "mediocre",
            "moderate", "neutral", "alright", "tolerable"
        }

        self.soft_positive_keywords = {
            "happy", "satisfied", "recommend", "brilliant",
            "delightful", "pleased", "content", "grateful",
            "memorable", "comfortable", "pleasant", "nice",
            "good", "clean", "polite", "friendly", "helpful",     "very friendly", "truly helpful", "highly recommend", 
    "extremely comfortable", "exceptionally smooth", "went above and beyond"
        }

        self.soft_negative_keywords = {
            "poor", "bad", "disappointing", "miserable",
            "unprofessional", "unsafe", "pathetic", "shocking",
            "lacking", "underwhelming", "not good", "could be better",
            "not ideal", "unfortunate", "not perfect", "bland",
            "crowded", "slow", "dated", "noisy", "overpriced",     "not worth it", "never again", "not recommended", "barely acceptable", 
    "not clean", "lackluster", "just okay", "meh", "mediocre at best"
        }

        self.soft_neutral_keywords = {
            "well organized", "organized", "hassle-free",
            "reliable", "supportive", "convenient", "basic",
            "small", "tight", "limited", "nothing special",
            "just fine", "it was ok", "no big deal",
            "as expected", "typical", "average at best",
            "nothing to complain about", "smooth", "acceptable",
            "simple", "fair"
        }

        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'hardly', 'scarcely', 
            'barely', 'doesn\'t', 'isn\'t', 'wasn\'t', 'shouldn\'t', 'wouldn\'t', 'couldn\'t', 'won\'t', 
            'can\'t', 'don\'t', 'rarely', 'seldom', 'not at all', 'no way', 'don\'t even', 'not quite', 
            'not really', 'not sure', 'hardly ever'
        }

        self.intensifiers = {
            'very': 1.1, 'extremely': 1.5, 'really': 1.3, 'super': 1.4,
            'incredibly': 1.8, 'absolutely': 1.7, 'totally': 1.6,
            'completely': 1.6, 'utterly': 1.9, 'exceptionally': 1.7
        }

        self.diminishers = {
            'slightly': 0.8, 'somewhat': 0.8, 'a bit': 0.8, 'kind of': 0.8,
            'sort of': 0.8, 'rather': 0.9, 'fairly': 0.9, 'relatively': 0.9
        }

        self.topics = {
            'staff': ['staff', 'employee', 'personnel', 'worker', 'crew', 'team'],
            'facilities': ['wifi', 'restroom', 'bathroom', 'toilet', 'shop', 'restaurant', 
                        'cafe', 'seating', 'chair', 'terminal', 'lounge', 'duty-free', 'gate', 
                        'concourse', 'area', 'zone', 'section', 'space', 'store', 'outlet', 
                        'food', 'beverage', 'snack'],
            'cleanliness': ['clean', 'dirty', 'filthy', 'hygiene', 'tidy', 'mess', 'maintenance', 
                            'sanitary', 'sanitation', 'garbage', 'trash', 'litter', 'smell', 'odor', 'stain'],
            'efficiency': ['queue', 'line', 'wait', 'delay', 'quick', 'fast', 'slow', 'efficient', 
                        'process', 'security check', 'boarding', 'disembarking', 'transfer', 
                        'connection', 'time', 'speed', 'pace', 'flow'],
            'transport': ['parking', 'bus', 'taxi', 'transport', 'connection', 'transfer', 
                        'accessibility', 'shuttle', 'train', 'metro', 'subway', 'car', 'vehicle', 
                        'transit', 'commute'],
            'comfort': ['comfortable', 'space', 'crowded', 'quiet', 'noisy', 'temperature', 
                        'air conditioning', 'seating', 'chair', 'bench', 'rest', 'relax', 'sleep', 
                        'nap', 'restroom', 'bathroom', 'toilet'],
            'staff_checkin': ['check-in', 'counter', 'agent', 'representative', 'reception', 'welcome', 
                            'assist', 'attendant', 'customer', 'queue', 'line', 'waiting', 'help', 
                            'support', 'clerk', 'desk', 'greeting'],
            'staff_security': ['security', 'officer', 'guard', 'screening', 'checkpoint', 'inspection', 
                            'patrol', 'boarding', 'bag check', 'x-ray', 'metal detector', 
                            'passport control', 'surveillance', 'safety', 'crowd control', 'escort', 'control'],
            'facilities_wifi': ['wifi', 'internet', 'connection', 'signal', 'speed', 'network', 'access', 
                                'hotspot', 'router', 'bandwidth', 'reception', 'reliable', 'secure', 'login', 
                                'free wifi', 'connection issues', 'slow internet', 'speed test', 'cafe', 'lounge', 
                                'waiting area'],
            'customer_service': ['support', 'helpdesk', 'customer service', 'assist', 'service desk', 
                                'call center', 'representative', 'complaint', 'resolution', 'solution', 
                                'contact', 'email support', 'phone support', 'response time'],
            'pricing': ['cost', 'price', 'value', 'affordable', 'expensive', 'discount', 'deal', 'budget', 
                        'cheap', 'overpriced', 'bargain', 'price point', 'value for money', 
                        'worth the cost', 'too expensive', 'too costly', 'underpriced', 'reasonable price'],
            'airline_experience': ['flight', 'seat', 'comfort', 'legroom', 'turbulence', 'boarding', 
                                'delayed', 'canceled', 'ticket', 'boarding process', 'takeoff', 'landing', 
                                'pilot', 'crew', 'passenger service', 'airline policies'] }

    def analyze_review(self, review_text):
        review_text = clean_text(review_text)
        topic_sentiments = defaultdict(int)
        topic_mentions = defaultdict(int)

        for topic, topic_words in self.topics.items():
            for word in topic_words:
                if word in review_text:
                    topic_mentions[topic] += 1
                    context = self._get_context(review_text, word)
                    sentiment = self._calculate_context_sentiment(context)
                    topic_sentiments[topic] += sentiment

        overall_sentiment = 'neutral'  # Default sentiment
        if any(topic_mentions.values()):  # If any topic is mentioned
            # Check for strong positive, negative, and soft positive/negative words
            if any(word in self.positive_keywords for word in review_text.split()):
                overall_sentiment = 'positive'
            elif any(word in self.negative_keywords for word in review_text.split()):
                overall_sentiment = 'negative'
            # Check soft positive/negative and adjust sentiment
            elif any(word in self.soft_positive_keywords for word in review_text.split()):
                overall_sentiment = 'positive'
            elif any(word in self.soft_negative_keywords for word in review_text.split()):
                overall_sentiment = 'negative'
            # Retain neutral if no strong/soft keywords are found
            elif any(word in self.soft_neutral_keywords for word in review_text.split()):
                overall_sentiment = 'neutral'

        return {
            'overall_sentiment': overall_sentiment,
            'topic_sentiments': dict(topic_sentiments),
            'topic_mentions': dict(topic_mentions)
        }

    def _calculate_context_sentiment(self, context_words):
        score = 0
        for word in context_words:
            if word in self.positive_keywords:
                score += 1
            elif word in self.negative_keywords:
                score -= 1
            elif word in self.neutral_keywords:
                score += 0
            # Adding soft positive/negative/neutral keywords to the score calculation
            elif word in self.soft_positive_keywords:
                score += 0.5  # Slight positive impact
            elif word in self.soft_negative_keywords:
                score -= 0.5  # Slight negative impact
            elif word in self.soft_neutral_keywords:
                score += 0  # No impact for neutral soft keywords
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

if __name__ == "__main__":
    input_file = "../../data/dataset/utest_set.csv"
    output_file = "../../data/keywordstopics/t_kt_results.csv"

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
                **result['topic_mentions'],
                **{f"{k}_sentiment": v for k, v in result['topic_sentiments'].items()}
            })

    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
