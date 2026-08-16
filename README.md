# Flight Experience Feedback Analysis

CSC 400 Research Project — Southern Connecticut State University
Authors: Joseph Rodriguez, Peter Chinedu Chukwu, Quinton C. Haughton, Mohamed Ramzeen, Shafaeat Hossain

## Paper

The full write-up is available here: [Flight Experience Feedback Analysis (PDF)](./CSC_400_Final_Research_Paper.pdf)

*Unpublished course research paper, CSC 400, Southern Connecticut State University.*

## Overview

This project analyzes airline passenger reviews using sentiment analysis to better understand customer satisfaction. It compares the effectiveness of three sentiment classification techniques on a large, hybrid labeled dataset scraped from airlinequality.com (Skytrax):

- **Word Score Sentiment Scoring (AFINN)** lexicon based word level scoring
- **Rule Based Sentiment Analysis (SentiWordNet)** POS aware lexicon scoring with negation and intensifier handling
- **Deep Learning Based Sentiment Analysis (cardiffnlp/twitter-xlm-roberta-base-sentiment)** transformer based classification

A hybrid labeling pipeline (TextBlob + VADER + keyword tagging, with DistilBERT pseudo labeling for ambiguous cases and GPT-4 generated synthetic neutral samples) was used to build a ground truth validation set, and each technique was evaluated on a held out test set using accuracy, precision, recall, and F1-score.

## Key Results

The word scoring (AFINN) approach achieved the strongest overall performance, slightly outperforming the deep learning model, while the rule-based method consistently performed the worst. This suggests that, with a well constructed labeled dataset, traditional lexicon-based methods can be competitive with more complex deep learning models.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Word Scoring (AFINN) | 80.8% | 0.81 | 0.81 | 0.81 |
| Deep Learning (XLM-RoBERTa) | 79.1% | 0.79 | 0.79 | 0.79 |
| Rule-Based (SentiWordNet) | 71.0% | 0.74 | 0.71 | 0.71 |

All three models excelled at identifying negative sentiment but struggled most with neutral sentiment, which was consistently the hardest class to classify across all methods.

## Dataset

- **Source:** 22,098 reviews scraped from airlinequality.com (Skytrax)
- **Fields:** reviewer name, date of submission, star rating (1-10), review text, sentiment label
- **Labeled Validation Set:** 12,000 reviews (4,000 per sentiment class: positive, neutral, negative)
- **Test Set:** 3,000 unused reviews, reserved for unbiased evaluation
- Cleaned dataset stored as `cleaned_skytrax_reviews.csv`

### Hybrid Labeling Pipeline

1. Initial labeling with TextBlob, VADER, and a custom keyword-based tagger
2. Reviews with unanimous (3/3) agreement accepted as high-confidence labels
3. Disagreements flagged and pseudo-labeled using `distilbert-base-uncased-finetuned-sst-2-english`
4. Synthetic neutral reviews generated with GPT-4 to balance the dataset
5. All labeled reviews merged into the final validation set; remaining reviews reserved for the test set

## Methodology Summary

- **Preprocessing:** verification label removal, URL/HTML stripping, special character cleaning, lowercase normalization, missing/empty review handling
- **Feature extraction:** POS tagging and negation-aware handling for the rule-based method, token-level AFINN scoring for the word-score method, subword tokenization and contextual embeddings for the transformer model
- **Train/test split:** 80/20, with 5-fold cross-validation
- **Evaluation:** accuracy, precision, recall, and F1-score per sentiment class, plus confusion matrices

## Setting Up the Virtual Environment (venv)

### Windows

```
cd path\to\your\project
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```
cd /path/to/your/project
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt once it's active.

## Installing Dependencies

```
pip install -r requirements.txt
```

Verify installation with:

```
pip list
```

Deactivate the environment when finished:

```
deactivate
```

## Key Libraries Used

| Library | Version | Purpose |
|---|---|---|
| beautifulsoup4 | 4.13.4 | Web scraping of airline passenger reviews |
| requests | 2.32.3 | HTTP requests to fetch review data |
| pandas | 2.2.3 | Data cleaning, manipulation, and analysis |
| nltk | 3.9.1 | Tokenization, POS tagging, and NLP utilities |
| textblob | 0.19.0 | Rule/lexicon-based sentiment scoring |
| vaderSentiment | 3.3.2 | Lexicon-based sentiment scoring for initial labeling |
| afinn | 0.1 | Word-score sentiment scoring |
| transformers | 4.51.2 | Pre-trained transformer models (XLM-RoBERTa, DistilBERT) |
| torch | 2.6.0 | Deep learning backend for transformer-based sentiment model |
| scikit-learn | 1.6.1 | Evaluation metrics (accuracy, precision, recall, F1) |
| openai | 0.28.0 | GPT-4 access for synthetic neutral review generation |
| python-dotenv | 1.1.0 | Managing API keys and environment configuration |
| matplotlib | 3.10.1 | Visualizations of sentiment distributions and results |
| seaborn | 0.13.2 | Statistical visualizations (heatmaps, comparison plots) |

See the paper's appendix for the full pinned dependency list.

## Experiment Environment

- **CPU:** AMD Ryzen 7 3700X (8 cores, 16 threads)
- **GPU:** NVIDIA GeForce RTX 2070 Super
- **RAM:** 32GB DDR4
- **OS:** Windows 10, 64-bit
- **Python:** 3.11.9

## Limitations

- Neutral sentiment was consistently the hardest class for all three models to classify correctly
- Rule-based and word-score methods struggle with sarcasm, negation nuance, and idiomatic language
- The deep learning model is more computationally expensive (~13 minutes on the validation set vs. under 30 seconds for the lexicon-based methods) and less interpretable
- Results are specific to the airline review domain and may not generalize without adaptation

## Future Work

- Improved neutral sentiment detection (contrastive learning, syntactic ambiguity detection)
- Hybrid ensemble models combining lexicon-based heuristics with transformer embeddings
- Domain adaptation and fine-tuning on airline-specific data
- Explainability techniques (SHAP, LIME, attention visualization) for the deep learning model
- Multilingual and cross-cultural sentiment analysis
- Real-time/scalable deployment via model compression or distillation

## Acknowledgment

This work was conducted at Southern Connecticut State University under the guidance of Professor MD Hossain.
