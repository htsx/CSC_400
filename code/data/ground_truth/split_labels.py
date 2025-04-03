import pandas as pd

# Load the dataset
file_path = "ground_truth_reviews_untouched.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# Split the dataset
labeled_df = df.iloc[:1001]  # First 1001 reviews are labeled
unlabeled_df = df.iloc[1001:]  # Remaining reviews are unlabeled

# Save to new CSV files
labeled_df.to_csv("labeled_reviews.csv", index=False)
unlabeled_df.to_csv("unlabeled_reviews.csv", index=False)

print("Dataset split complete! Labeled and unlabeled data saved separately.")