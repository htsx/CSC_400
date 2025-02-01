import csv

#clean the data
def clean_airport_ratings(filepath):
    cleaned_data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  #header row
        cleaned_data.append(header + ['cleaned_text'])  #new column for cleaned text

        for row in reader:
            # Drop rows with missing ratings
            if not row[1]:  # Assuming 'Rating' is the second column
                continue

            # Clean the 'Rating' column by stripping spaces
            cleaned_text = row[1].strip()
            cleaned_data.append(row + [cleaned_text])

    #save cleaned data
    cleaned_filepath = 'cleaned_reviews.csv'
    with open(cleaned_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cleaned_data)

    print(f"Cleaned data saved to {cleaned_filepath}")
    return cleaned_filepath

#add sentiment scores (implemented later)
def add_sentiment_scores(filepath):
    data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  #header row
        data.append(header + ['sentiment_score'])  #new column for sentiment scores

        for row in reader:
            #for sentiment analysis (implemented later)
            sentiment_score = 0.0  #default value
            data.append(row + [sentiment_score])

    #save the results
    output_path = 'data/sentiment_reviews.csv'
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"Sentiment scores saved to {output_path}")

#main 
if __name__ == "__main__":
    #clean the data
    cleaned_data_path = clean_airport_ratings('data/airport_ratings.csv')
    
    #add sentiment scores ()
    add_sentiment_scores(cleaned_data_path)