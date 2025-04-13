import pandas as pd

# Load the balanced CSV file
df = pd.read_csv("../../data/webscrapper/sampled.csv")

# Check the first few rows to understand the structure
print(df.head())

# Clean the review_text column by:
# 1. Ensuring the review_text is a string (convert any non-string types to string)
# 2. Removing rows with missing or invalid review texts
df['review_text'] = df['review_text'].apply(lambda x: str(x) if not isinstance(x, str) else x)  # Convert to string if not already
df = df[df['review_text'].notna() & (df['review_text'].str.strip() != '')]  # Remove rows with empty or NaN review_text

# Optionally, log or inspect how many rows were removed
print(f"Rows before cleaning: {len(df)}")

# Add an empty 'review_classification' column that will hold the predicted sentiment labels
df['review_classification'] = None  # Create an empty column to store the predicted sentiment labels

# Save the cleaned and updated data to a new file
df.to_csv("../../data/webscrapper/unlabeled_reviews1.csv", index=False)

print("Cleaned and updated dataset with empty 'review_classification' column saved to 'unlabeled_reviews.csv'")
