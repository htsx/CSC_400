import pandas as pd

# Load datasets
unused_df = pd.read_csv('data/unused_reviews.csv')

# Filter only those under 512 characters
unused_df = unused_df[unused_df['review_text'].astype(str).str.len() < 512].copy()

# Separate by true classification
pos_all = unused_df[unused_df['review_classification'] == 'Positive'].copy()
neg_all = unused_df[unused_df['review_classification'] == 'Negative'].copy()

# Agreement logic functions
def agreement_level(row, sentiment):
    votes = [
        row['hybrid_sentiment'],
        row['textblob_label'],
        row['vader_label'],
        row['keyword_label']
    ]
    return votes.count(sentiment)

def select_reviews(df, sentiment, target_count):
    # Priority order: 4/4 → 3/4 → 2/4 → hybrid only
    selected = pd.DataFrame()
    used_idx = set()

    for threshold, label in [(4, '4/4'), (3, '3/4'), (2, '2/4')]:
        candidates = df.loc[~df.index.isin(used_idx)].copy()
        candidates = candidates[candidates.apply(lambda x: agreement_level(x, sentiment) == threshold, axis=1)]
        candidates['label_agreement_level'] = label

        needed = target_count - len(selected)
        if needed <= 0:
            break

        if len(candidates) > needed:
            candidates = candidates.sample(n=needed, random_state=42)
        selected = pd.concat([selected, candidates], ignore_index=True)
        used_idx.update(candidates.index)

    # Fallback to hybrid sentiment only
    if len(selected) < target_count:
        needed = target_count - len(selected)
        fallback = df.loc[~df.index.isin(used_idx)].copy()
        fallback = fallback[fallback['hybrid_sentiment'] == sentiment]
        fallback['label_agreement_level'] = '1/4-hybrid'
        if len(fallback) > needed:
            fallback = fallback.sample(n=needed, random_state=42)
        selected = pd.concat([selected, fallback], ignore_index=True)

    # Ensure we have enough reviews by sampling from available data if target is not met
    if len(selected) < target_count:
        remaining_needed = target_count - len(selected)
        additional = df.loc[~df.index.isin(used_idx)].sample(n=remaining_needed, random_state=42)
        selected = pd.concat([selected, additional], ignore_index=True)

    return selected

# Select reviews
POS_TARGET = 1000
NEG_TARGET = 1000
final_pos = select_reviews(pos_all, 'Positive', POS_TARGET)
final_neg = select_reviews(neg_all, 'Negative', NEG_TARGET)

# Load neutral reviews
neutral_df = pd.read_csv('data/generated_1000_neutral_reviews.csv')
neutral_df = neutral_df.rename(columns={'text': 'review_text'})
neutral_df = neutral_df[neutral_df['review_text'].astype(str).str.len() < 512].copy()
neutral_df['review_classification'] = 'Neutral'
neutral_df['label_agreement_level'] = 'Generated'
neutral_df = neutral_df.sample(n=1000, random_state=42)

# Combine all
final_test_set = pd.concat([final_pos, final_neg, neutral_df], ignore_index=True)

# Final sanity check
print("✅ Final test set counts:")
print("   Positive:", len(final_pos))
print("   Negative:", len(final_neg))
print("   Neutral :", len(neutral_df))
print("   Total   :", len(final_test_set))

# Save to file
final_test_set.to_csv('../../data/dataset/test_set.csv', index=False)
print("\n✅ Saved final test set to 'test_set.csv'")
