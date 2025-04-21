import pandas as pd

df = pd.read_csv('../../data/dataset/test_set.csv')  # Path to your sampled CSV

#Set the 'review_classification' column to 'Unlabeled' or NaN
df['review_classification'] = 'Unlabeled'

#Save the modified DataFrame with unlabeled classifications
df.to_csv('../../data/dataset/utest_set.csv', index=False)

print("The review classifications have been set to 'Unlabeled' for the 300 reviews and saved.")
