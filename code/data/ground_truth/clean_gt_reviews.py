import pandas as pd

# Load the ground truth reviews CSV file
df_truth = pd.read_csv("ground_truth_reviews.csv")

# Limit the data to the first 1001 rows
df_truth_manually_labeled = df_truth.head(1001)

# Remove leading/trailing spaces and standardize to uppercase
df_truth_manually_labeled['ground_truth_sentiment'] = df_truth_manually_labeled['ground_truth_sentiment'].str.strip().str.upper()

# Check for NaN values specifically in the 'ground_truth_sentiment' column for the first 1001 rows
nan_count = df_truth_manually_labeled['ground_truth_sentiment'].isna().sum()

# Print the count of NaN values in the 'ground_truth_sentiment' column
print(f"There are {nan_count} NaN values in the 'ground_truth_sentiment' column for the first 1001 rows.")

# If there are NaN values, print the rows with NaN values
if nan_count > 0:
    nan_reviews = df_truth_manually_labeled[df_truth_manually_labeled['ground_truth_sentiment'].isna()]
    print("\nRows with NaN in 'ground_truth_sentiment':")
    print(nan_reviews)
    
    # Manually replace the NaN value at row 1000 (or handle other NaN values)
    df_truth_manually_labeled.loc[df_truth_manually_labeled['ground_truth_sentiment'].isna(), 'ground_truth_sentiment'] = 'NEGATIVE'
    
    # Save the cleaned data back to the same file
    df_truth_manually_labeled.to_csv("ground_truth_reviews.csv", index=False)
    print("\nNaN values replaced, and cleaned data saved to 'ground_truth_reviews.csv'.")
    
    # Recheck for NaN values after replacing
    nan_count_after = df_truth_manually_labeled['ground_truth_sentiment'].isna().sum()
    print(f"\nAfter replacing NaN values, there are {nan_count_after} NaN values remaining in the 'ground_truth_sentiment' column.")
else:
    print("\nNo NaN values found in the 'ground_truth_sentiment' column for the first 1001 rows.")
