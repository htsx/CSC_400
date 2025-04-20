import pandas as pd
import re

# Load the original generated file
df = pd.read_csv("openai_generated_neutral_reviews.csv")

# Function to clean review text
def clean_review(text):
    if pd.isna(text):
        return ""
    # Remove leading numbers like "6. " or "12. "
    text = re.sub(r"^\s*\d+\.\s*", "", text)

    # Remove all leading/trailing quotes and whitespace
    text = text.strip().strip('"').strip("'").strip()

    # Replace escaped double quotes (e.g., \" or double-double quotes "")
    text = re.sub(r'\\?"{1,2}', '', text)

    return text

# Apply the cleaning function
df["review_text"] = df["review_text"].apply(clean_review)

# Save the cleaned CSV
df.to_csv("openai_generated_neutral_reviews_cleaned.csv", index=False)
print("✅ Fully cleaned review_text and saved to openai_generated_neutral_reviews_cleaned.csv")
