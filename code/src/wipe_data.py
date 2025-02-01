import os

# List of files to wipe clean
files_to_wipe = [
    'data/airport_ratings.csv',
    'data/cleaned_reviews.csv',
    'data/sentiment_reviews.csv'
]

def wipe_files():
    for file in files_to_wipe:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")
        else:
            print(f"{file} does not exist. Skipping...")

if __name__ == "__main__":
    wipe_files()