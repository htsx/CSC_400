import csv

#function for cleaning text
def clean_text(text):
    #text cleaning (to be implemented later)
    return text

#main script
if __name__ == "__main__":
    input_path = 'data/airport_ratings.csv'
    output_path = 'data/cleaned_reviews.csv'

    #read the input CSV file
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  #header row
        rows = [row for row in reader]

    #add a new column for cleaned text
    header.append('cleaned_text')

    #clean the text column (placeholder for now)
    cleaned_rows = []
    for row in rows:
        cleaned_text = clean_text(row[1])
        cleaned_rows.append(row + [cleaned_text])

    #write the cleaned data to a new CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  #header
        writer.writerows(cleaned_rows)  #cleaned rows

    print(f"Cleaned data saved to {output_path}")