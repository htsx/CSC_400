import pandas as pd
import random

# Load the already labeled reviews data
ground_truth_reviews = pd.read_csv('../../data/webscrapper/ground_truth_reviews.csv')  # Replace with your file path

# Get the current neutral reviews
neutral_reviews = ground_truth_reviews[ground_truth_reviews['review_classification'] == 'Neutral']

# Filter out reviews with missing or empty text
neutral_reviews = neutral_reviews[neutral_reviews['review_text'].notna() & (neutral_reviews['review_text'].str.strip() != '')]

# Check how many neutral reviews you currently have
current_neutral_count = len(neutral_reviews)
required_neutral_count = 4000  # You need 4000 neutral reviews
reviews_to_generate = required_neutral_count - current_neutral_count

print(f"Currently, you have {current_neutral_count} neutral reviews.")
print(f"You need to generate {reviews_to_generate} more neutral reviews.")

def make_neutral(review_text):
    # Expanded list of neutral words for variety
    neutral_words = {
        'great': ['adequate', 'satisfactory', 'fine', 'okay', 'acceptable'],
        'awful': ['subpar', 'unsatisfactory', 'mediocre', 'average', 'below expectations'],
        'fantastic': ['acceptable', 'reasonable', 'fine', 'good enough', 'pleasant'],
        'horrible': ['unremarkable', 'passable', 'average', 'lackluster', 'not great'],
        'amazing': ['fine', 'decent', 'pleasant', 'adequate', 'okay'],
        'disappointing': ['unsurprising', 'predictable', 'average', 'underwhelming'],
        'wonderful': ['decent', 'adequate', 'fair', 'satisfactory'],
        'terrible': ['suboptimal', 'unremarkable', 'below average', 'not ideal'],
        'love': ['like', 'enjoy', 'prefer', 'appreciate'],
        'hate': ['dislike', 'prefer less', 'don’t favor'],
        'always': ['often', 'frequently', 'regularly', 'occasionally'],
        'never': ['rarely', 'infrequently', 'seldom'],
        'lacklustre': ['adequate', 'mediocre', 'ordinary', 'unimpressive'],
        'bizarre': ['unusual', 'unexpected', 'strange', 'peculiar'],
        'tedious': ['routine', 'mundane', 'standard', 'monotonous'],
        'dried': ['well-cooked', 'overdone', 'well-prepared'],
        'missed': ['overlooked', 'ignored', 'neglected'],
        'too small': ['smaller than expected', 'more compact', 'more limited'],
        'slow': ['delayed', 'leisurely', 'gradual', 'not fast enough'],
        'uncomfortable': ['less comfortable', 'unpleasant', 'slightly uncomfortable', 'inconvenient'],
        'poor': ['subpar', 'below average', 'not great'],
        'unresponsive': ['slow to respond', 'delayed response', 'unavailable'],
        'unsatisfactory': ['not as expected', 'below expectations', 'lacking'],
        'inconvenient': ['challenging', 'difficult', 'not ideal', 'inaccessible']
    }

    # Neutral phrases for additional variety
    neutral_phrases = {
        'only good at finding excuses': 'the company has faced challenges with delays',
        'always late': 'there have been repeated delays',
        'nobody to unload luggage': 'there were delays in unloading luggage',
        'no stairs no bus': 'there were no stairs or buses available at arrival',
        'putting its passengers into trouble': 'this led to some inconveniences for passengers',
        'complained about the lack of service': 'there were some concerns raised about the service quality',
        'very late': 'there were significant delays',
        'terrible experience': 'the experience was less than expected'
    }

    # Replacing negative or emotional language with more neutral terms
    review_text_lower = review_text.lower()

    # Replace strong adjectives with neutral alternatives, randomly selecting from lists
    for word, neutral_list in neutral_words.items():
        if word in review_text_lower:
            replacement = random.choice(neutral_list)  # Randomly pick a neutral word
            review_text_lower = review_text_lower.replace(word, replacement)

    # Replace emotionally charged phrases with neutral ones
    for phrase, neutral_phrase in neutral_phrases.items():
        if phrase in review_text_lower:
            review_text_lower = review_text_lower.replace(phrase, neutral_phrase)

    # Modify sentence structure and use softer language
    review_text_lower = review_text_lower.replace('very', 'somewhat')
    review_text_lower = review_text_lower.replace('extremely', 'quite')
    review_text_lower = review_text_lower.replace('best', 'acceptable')
    review_text_lower = review_text_lower.replace('worst', 'suboptimal')
    
    # Adding variety with sentence restructuring (example: converting "always late" to "there are delays often")
    review_text_lower = review_text_lower.replace('always late', 'delays are often reported')

    # Return the modified neutral review text
    return review_text_lower

# Generate new neutral reviews
new_reviews = []
for _ in range(reviews_to_generate):
    random_review = neutral_reviews.sample(n=1)['review_text'].values[0]
    new_review = make_neutral(random_review)
    new_reviews.append(new_review)

# Convert the new reviews into a DataFrame
new_reviews_df = pd.DataFrame({'review_text': new_reviews, 'review_classification': 'Neutral'})

# Add missing columns with placeholder values
new_reviews_df['review_name'] = 'Unknown'  # Placeholder value, replace as necessary
new_reviews_df['review_type'] = 'General'  # Placeholder value, replace as necessary
new_reviews_df['passenger_name'] = 'Anonymous'  # Placeholder value, replace as necessary
new_reviews_df['review_date'] = pd.NaT  # NaT represents Not a Time (missing date)

# Append the new reviews to the original dataset
augmented_reviews = pd.concat([ground_truth_reviews, new_reviews_df], ignore_index=True)

# Save the augmented dataset back to a CSV file
augmented_reviews.to_csv('with_more_neutral.csv', index=False)

print(f"{reviews_to_generate} new neutral reviews have been generated and saved.")
