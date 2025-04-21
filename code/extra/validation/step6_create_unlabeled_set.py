import pandas as pd

df = pd.read_csv('../../data/dataset/validation_set.csv')  # Path to your sampled CSV

#Set the 'review_classification' column to 'Unlabeled' or NaN
df['review_classification'] = 'Unlabeled'

#Save the modified DataFrame with unlabeled classifications
df.to_csv('../../data/dataset/uvalidation_set.csv', index=False)

print("The review classifications have been set to 'Unlabeled' for the reviews and saved.")
