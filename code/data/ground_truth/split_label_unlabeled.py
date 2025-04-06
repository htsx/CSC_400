import pandas as pd

# Load the dataset
file_path = "ground_truth_reviews_untouched.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# Extract only the unlabeled reviews (after the first 1001 rows)
unlabeled_df = df.iloc[1001:]

# Save to a new CSV file
unlabeled_df.to_csv("unlabeled_reviews.csv", index=False)

print("Unlabeled reviews saved to 'unlabeled_reviews.csv'.")
