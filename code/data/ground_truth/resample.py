import pandas as pd
from sklearn.utils import resample

# --- CONFIGURATION ---
input_file = "ground_truth_reviews.csv"  # Your 1001 human-annotated reviews
output_file = "balanced_ground_truth.csv"  # Output balanced file
sentiment_col = "ground_truth_sentiment"  # Column with human labels

# Target class sizes (adjust based on your needs)
target_counts = {
    "NEGATIVE": 400,  # Original: 640
    "POSITIVE": 300,  # Original: 256
    "NEUTRAL": 301    # Total should be 1001 (adjusted from 105)
}

# --- LOAD DATA ---
print(f"Loading human-annotated data from {input_file}...")
df = pd.read_csv(input_file)

# Verify required columns exist
assert sentiment_col in df.columns, f"Column '{sentiment_col}' not found!"
assert "review_text" in df.columns, "Review text column missing!"

# --- HYBRID RESAMPLING ---
print("\nOriginal class distribution:")
print(df[sentiment_col].value_counts())

balanced_dfs = []
for class_name, target_count in target_counts.items():
    class_df = df[df[sentiment_col] == class_name]
    
    if len(class_df) > target_count:
        # Undersample majority class (random selection)
        resampled = class_df.sample(n=target_count, random_state=42)
    else:
        # Oversample minority classes (with replacement)
        resampled = resample(class_df, 
                           replace=True, 
                           n_samples=target_count, 
                           random_state=42)
    balanced_dfs.append(resampled)

# Combine and shuffle
balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42)

# --- SAVE BALANCED DATA ---
print("\nBalanced distribution:")
print(balanced_df[sentiment_col].value_counts())

balanced_df.to_csv(output_file, index=False)
print(f"\n✅ Saved balanced ground truth to {output_file}")
