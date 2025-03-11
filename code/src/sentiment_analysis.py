import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import nltk

# Ensure necessary nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the reviews dataset
reviews_df = pd.read_csv('data/cleaned_skytrax_reviews.csv')

# Sentiment Scoring & Distribution Analysis using TextBlob and VADER
def sentiment_scoring(review_text):
    # TextBlob Sentiment
    textblob_sentiment = TextBlob(review_text).sentiment.polarity

    # VADER Sentiment
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_sentiment = vader_analyzer.polarity_scores(review_text)['compound']

    return textblob_sentiment, vader_sentiment

# Apply sentiment analysis to reviews
reviews_df[['textblob_sentiment', 'vader_sentiment']] = reviews_df['review_text'].apply(
    lambda text: pd.Series(sentiment_scoring(str(text)))
)

# Keyword & Topic-Based Sentiment Analysis
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.isalpha() and word not in stop_words]

# Preprocess all reviews
reviews_df['processed_text'] = reviews_df['review_text'].apply(lambda text: preprocess_text(str(text)))

# Create a dictionary and corpus for topic modeling
dictionary = corpora.Dictionary(reviews_df['processed_text'])
corpus = [dictionary.doc2bow(text) for text in reviews_df['processed_text']]

# Train an LDA model
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# Save the updated dataframe
reviews_df.to_csv('data/processed_reviews.csv', index=False)

print("Sentiment analysis and topic modeling complete. Processed data saved.")
