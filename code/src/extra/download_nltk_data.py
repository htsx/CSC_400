# download_nltk_data.py
import nltk

def download_nltk_data():
    nltk.download('punkt')  # For tokenization
    nltk.download('stopwords')  # For stopword removal
    nltk.download('punkt_tab')  # For additional tokenizer data

if __name__ == "__main__":
    download_nltk_data()