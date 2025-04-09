from collections import defaultdict
import re

class ReviewAnalyzer:
    def __init__(self):
        # Expanded airport-specific sentiment keywords
        self.positive_keywords = {
            'excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic',
            'love', 'perfect', 'best', 'awesome', 'happy', 'satisfied',
            'recommend', 'helpful', 'impressed', 'outstanding', 'friendly',
            'clean', 'efficient', 'quick', 'modern', 'convenient', 'easy',
            'comfortable', 'smooth', 'pleasant', 'nice', 'airy', 'brilliant',
            'exceptional', 'superb', 'terrific', 'delightful', 'enjoyable',
            'satisfactory', 'pleased', 'content', 'grateful', 'appreciate',
            'welcoming', 'professional', 'courteous', 'polite', 'attentive',
            'spacious', 'well-maintained', 'organized', 'streamlined', 'hassle-free'
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
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither',
            'nowhere', 'hardly', 'scarcely', 'barely', 'doesn\'t', 'isn\'t',
            'wasn\'t', 'shouldn\'t', 'wouldn\'t', 'couldn\'t', 'won\'t',
            'can\'t', 'don\'t'
        }
        
        # Intensifier words
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.3, 'super': 1.4,
            'incredibly': 1.8, 'absolutely': 1.7, 'totally': 1.6,
            'completely': 1.5, 'utterly': 1.9, 'exceptionally': 1.7
        }
        
        # Diminisher words
        self.diminishers = {
            'slightly': 0.7, 'somewhat': 0.8, 'a bit': 0.8, 'kind of': 0.8,
            'sort of': 0.8, 'rather': 0.9, 'fairly': 0.9, 'relatively': 0.9
        }
        
        # Define airport-specific topics and their associated words
        self.topics = {
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
                       'bathroom', 'toilet']
        }

    def analyze_review(self, review_text):
        # Convert review to lowercase for better matching
        review_text = review_text.lower()
        
        # Handle negation and context
        sentiment_score = self._calculate_sentiment_score(review_text)
        
        # Analyze topics with improved context
        topic_sentiments = defaultdict(int)
        topic_mentions = defaultdict(int)
        
        # Look for topics and nearby sentiment words with improved context
        for topic, topic_words in self.topics.items():
            for word in topic_words:
                if word in review_text:
                    topic_mentions[topic] += 1
                    context = self._get_context(review_text, word)
                    sentiment = self._calculate_context_sentiment(context)
                    topic_sentiments[topic] += sentiment

        # Determine overall sentiment with improved thresholds
        if sentiment_score >= 3:  # Increased threshold for positive
            overall_sentiment = "positive"
        elif sentiment_score <= -3:  # Increased threshold for negative
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
        """Calculate sentiment score with negation and intensifier handling"""
        words = text.split()
        score = 0
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.negation_words:
                # Look ahead for sentiment words
                for j in range(i+1, min(i+4, len(words))):
                    if words[j] in self.positive_keywords:
                        score -= 1
                    elif words[j] in self.negative_keywords:
                        score += 1
                continue
                
            # Check for intensifiers
            if word in self.intensifiers:
                # Look ahead for sentiment words
                for j in range(i+1, min(i+3, len(words))):
                    if words[j] in self.positive_keywords:
                        score += self.intensifiers[word]
                    elif words[j] in self.negative_keywords:
                        score -= self.intensifiers[word]
                continue
                
            # Check for diminishers
            if word in self.diminishers:
                # Look ahead for sentiment words
                for j in range(i+1, min(i+3, len(words))):
                    if words[j] in self.positive_keywords:
                        score += self.diminishers[word]
                    elif words[j] in self.negative_keywords:
                        score -= self.diminishers[word]
                continue
            
            # Basic sentiment scoring
            if word in self.positive_keywords:
                score += 1
            elif word in self.negative_keywords:
                score -= 1
                
        return score

    def _calculate_context_sentiment(self, context_words):
        """Calculate sentiment score for a context window"""
        score = 0
        for word in context_words:
            if word in self.positive_keywords:
                score += 1
            elif word in self.negative_keywords:
                score -= 1
        return score

    def _get_context(self, text, word, window_size=7):
        """Get words around a specific word in the text with larger window"""
        words = text.split()
        try:
            word_index = words.index(word)
            start = max(0, word_index - window_size)
            end = min(len(words), word_index + window_size + 1)
            return set(words[start:end])
        except ValueError:
            return set() 