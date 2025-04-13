import pandas as pd

# Load the review data from the CSV file
reviews_data = pd.read_csv("../../data/webscrapper/balanced.csv")

# Check if the 'sentiment' column exists
if 'review_classification' not in reviews_data.columns:
    print("Error: The CSV file needs a 'sentiment' column.")
else:
    # Count the number of positive, neutral, and negative reviews
    sentiment_counts = reviews_data['review_classification'].value_counts()

    # Display the counts
    print("Sentiment Review Counts:")
    print(sentiment_counts)
