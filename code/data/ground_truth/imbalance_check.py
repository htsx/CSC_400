import pandas as pd
df = pd.read_csv("ground_truth_reviews.csv")
print(df["ground_truth_sentiment"].value_counts())