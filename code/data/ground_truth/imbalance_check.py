import pandas as pd
df = pd.read_csv("balanced_ground_truth.csv")
print(df["ground_truth_sentiment"].value_counts())