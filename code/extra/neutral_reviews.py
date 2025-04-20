import openai
import pandas as pd
import time
from dotenv import OPENAI_API_KEY

#Set API key
openai.api_key = OPENAI_API_KEY
target_count = 4000
output_file = "openai_generated_neutral_reviews.csv"
batch_size = 20  # Number of reviews to generate per API call

#Prompt to generate realistic neutral airline reviews
system_prompt = "You are a professional reviewer generating realistic airline reviews."
user_prompt_template = (
    "Generate {} short, neutral airline reviews. "
    "Each review should be 1–3 sentences long, describe a typical or average experience, "
    "and avoid emotional or strongly positive/negative language."
)

#Storage
all_reviews = []

#Loop to generate batches
while len(all_reviews) < target_count:
    try:
        user_prompt = user_prompt_template.format(batch_size)

        #Use the correct method: ChatCompletion.create()
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{
                "role": "system", "content": system_prompt
            }, {
                "role": "user", "content": user_prompt
            }],
            temperature=0.7
        )

        raw_output = response['choices'][0]['message']['content']
        reviews = [line.strip("- ").strip() for line in raw_output.split("\n") if line.strip()]
        all_reviews.extend(reviews)
        print(f"Generated {len(all_reviews)} / {target_count}")

        time.sleep(1.2)  #API rate limits

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)

#Fnal trimming if overshot
all_reviews = all_reviews[:target_count]

#Create DataFrame and fill required columns
neutral_df = pd.DataFrame({
    "review_text": all_reviews,
    "review_classification": "Neutral",
    "review_name": "Unknown",
    "review_type": "General",
    "passenger_name": "Anonymous",
    "review_date": pd.NaT,
    "source": "synthetic"
})

#Save to CSV
neutral_df.to_csv(output_file, index=False)
print(f"Saved {len(neutral_df)} neutral reviews to {output_file}")
