import nltk

# List of all necessary NLTK resources
resources = [
    'punkt',  # For tokenization
    'averaged_perceptron_tagger',  # For POS tagging
    'wordnet',  # WordNet corpus
    'sentiwordnet',  # SentiWordNet corpus for sentiment analysis
    'averaged_perceptron_tagger_eng',  # English POS Tagger
]

# Download each resource
for resource in resources:
    try:
        nltk.download(resource)
        print(f"Downloaded {resource}")
    except Exception as e:
        print(f"Error downloading {resource}: {e}")
