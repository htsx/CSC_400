import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sentiment_analyzer import ReviewAnalyzer

def load_ground_truth(file_path):
    """Load ground truth data from CSV file"""
    return pd.read_csv(file_path)

def map_sentiment_to_numeric(sentiment):
    """Map sentiment labels to numeric values for evaluation"""
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    return sentiment_map.get(str(sentiment).lower(), 0)

def evaluate_sentiment_analysis(ground_truth_path):
    """
    Evaluate sentiment analysis performance against ground truth data
    using Accuracy, Precision, Recall, and F1 Score metrics
    """
    # Load ground truth data
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Print column names for debugging
    print("Available columns:", ground_truth.columns.tolist())
    
    # Initialize sentiment analyzer
    analyzer = ReviewAnalyzer()
    
    # Get predictions
    predictions = []
    for review in ground_truth['review_text']:  # Using 'review_text' column for review text
        # Convert review to string if it's not already
        review_str = str(review) if pd.notna(review) else ""
        result = analyzer.analyze_review(review_str)
        predictions.append(map_sentiment_to_numeric(result['overall_sentiment']))
    
    # Get ground truth labels - using hybrid_sentiment as the ground truth
    true_labels = ground_truth['hybrid_sentiment'].apply(map_sentiment_to_numeric)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Score': [accuracy, precision, recall, f1]
    })
    
    # Print results
    print("\nSentiment Analysis Evaluation Results:")
    print("=" * 50)
    print(results.to_string(index=False))
    print("=" * 50)
    
    # Save results to CSV
    results.to_csv('sentiment_evaluation_results.csv', index=False)
    print("\nResults saved to 'sentiment_evaluation_results.csv'")
    
    return results

if __name__ == "__main__":
    # Path to ground truth data - using the correct path
    ground_truth_path = "code/data/webscrapper/ground_truth_reviews.csv"
    
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        print("Please ensure the ground truth file exists in the correct location.")
    else:
        evaluate_sentiment_analysis(ground_truth_path) 