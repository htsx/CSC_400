import pandas as pd
from faker import Faker
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# Load the dataset
input_file = '../../data/webscrapper/ground_truth_reviews.csv'  # Path to your original reviews dataset
output_file = '../../data/webscrapper/balanced_ground_truth.csv'  # Path to save the new balanced dataset

# Load data
df = pd.read_csv(input_file)

# Handle missing values: replace NaNs in 'review_text' with an empty string
df['review_text'].fillna('', inplace=True)

# Initialize Faker to generate fake names
fake = Faker()

# Function to generate fake passenger names
def generate_fake_name():
    return fake.name()

# Function to generate fake review date (within the past year)
def generate_fake_date():
    return fake.date_this_year()

# Prepare the features for SMOTE (just the review text and classification)
X = df['review_text']
y = df['review_classification']

# Vectorize the review text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # You can change the number of features as needed
X_tfidf = vectorizer.fit_transform(X)

# Create an instance of SMOTE to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Rebuild the DataFrame with resampled data
resampled_df = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
resampled_df['review_classification'] = y_resampled

# Adding fake names, real airport names, and dates back to the dataframe
resampled_df['review_name'] = df['review_name'].sample(n=len(resampled_df), replace=True).values
resampled_df['passenger_name'] = resampled_df.apply(lambda x: generate_fake_name(), axis=1)
resampled_df['review_date'] = resampled_df.apply(lambda x: generate_fake_date(), axis=1)

# Add the rating value back (sampled from the original dataset)
resampled_df['rating_value'] = df['rating_value'].sample(n=len(resampled_df), replace=True).values

# Save the final balanced dataset without sentiment labels
resampled_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Balanced dataset with fake names saved to {output_file}")
