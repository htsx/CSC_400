import pandas as pd

# Load your full labeled dataset
df = pd.read_csv("../../data/webscrapper/cleaned_balanced.csv")  # Change path if needed

# Check the column name that contains the labels (adjust if needed)
label_column = 'review_classification'  # or 'hybrid_sentiment' depending on your file

# Sample 1000 from each label
sampled_df = (
    df.groupby(label_column, group_keys=False)
    .apply(lambda x: x.sample(n=1000, random_state=42))
    .sample(frac=1, random_state=42)  # Shuffle the final DataFrame
)

# Save the result
sampled_df.to_csv("../../data/webscrapper/sampled2.csv", index=False)
print("✅ Sampled 3,000 reviews (1,000 per label) saved to 'sampled_3000_reviews.csv'")
