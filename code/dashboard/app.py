import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='../templates')

def load_and_merge_data():
    # Load the cleaned reviews and sentiment scores
    cleaned_data = pd.read_csv('data/cleaned_reviews.csv')
    sentiment_data = pd.read_csv('data/sentiment_reviews.csv')

    # Print the columns and the first few rows to inspect the data
    print("Columns in cleaned data:", cleaned_data.columns)
    print("Columns in sentiment data:", sentiment_data.columns)
    print("First few rows of cleaned data:\n", cleaned_data.head())
    print("First few rows of sentiment data:\n", sentiment_data.head())

    # Ensure that 'Airport' contains the actual airport names from 'cleaned_text'
    cleaned_data['Airport'] = cleaned_data['cleaned_text'].apply(lambda x: x if isinstance(x, str) else "")
    sentiment_data['Airport'] = sentiment_data['cleaned_text'].apply(lambda x: x if isinstance(x, str) else "")

    # Convert 'Airport' columns to strings
    cleaned_data['Airport'] = cleaned_data['Airport'].astype(str)
    sentiment_data['Airport'] = sentiment_data['Airport'].astype(str)

    # Merge the two dataframes on 'Airport'
    merged_data = pd.merge(cleaned_data, sentiment_data, on='Airport', how='left', suffixes=('_cleaned', '_sentiment'))

    return merged_data

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    message = None
    search_query = None
    data = None

    if request.method == 'POST':
        search_query = request.form.get('search_query', '').strip().lower()  # Safe retrieval of form data

        # Handle the case where no search query is provided
        if not search_query:
            message = "Please enter an airport name to search."
        else:
            # Load and merge data
            merged_data = load_and_merge_data()

            # Convert the 'Airport' column to lowercase for case-insensitive search
            merged_data['Airport'] = merged_data['Airport'].str.lower()

            # Print all airport names for debugging
            print("Search Query:", search_query)
            print("Merged Data 'Airport' Column:")
            print(merged_data['Airport'].head(20))  # Check first 20 rows of merged data for debugging

            # Perform a case-insensitive search
            data = merged_data[merged_data['Airport'].str.contains(search_query, na=False)]

            # Print search results for debugging
            print("Search Results (first 5):")
            print(data.head())

            if data.empty:
                message = f"No results found for '{search_query.upper()}'."
            else:
                message = f"Results for '{search_query.upper()}'"

    return render_template('index.html', data=data, message=message, search_query=search_query)


if __name__ == "__main__":
    app.run(debug=True)
