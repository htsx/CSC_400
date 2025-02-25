import csv
import re

#function to clean the review text by removing verification strings
def clean_review_text(review_text):
    return re.sub(r"(✅Verified Review\| |✅Trip Verified\| |Not Verified\| )", "", review_text)

#function to filter and clean the raw reviews data
def filter_reviews(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        writer.writeheader()
        for row in reader:
            row['review_text'] = clean_review_text(row['review_text'])
            writer.writerow(row)

    print(f"Cleaned reviews saved to {output_file}")

if __name__ == "__main__":
    #raw review csv file path
    raw_reviews_file = 'data/raw_reviews.csv'
    #cleaned review csv file path
    cleaned_reviews_file = 'data/cleaned_reviews.csv'

    #filter and clean the raw reviews csv file
    filter_reviews(raw_reviews_file, cleaned_reviews_file)