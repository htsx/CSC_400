import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_airport_ratings(url):
    """
    Scrapes airport names and ratings from the Skytrax A-Z of Airport Ratings page.

    Args:
        url (str): The URL of the page to scrape.

    Returns:
        list: A list of dictionaries with airport names and ratings.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    # Locate the table containing the ratings (modify selectors as needed for the website structure)
    ratings_table = soup.find('table')  # Replace with a more specific selector if possible
    if not ratings_table:
        print("No table found on the page.")
        return []

    # Extract data from the table rows
    airports = []
    for row in ratings_table.find_all('tr')[1:]:  # Skip header row
        columns = row.find_all('td')
        if len(columns) >= 2:  # Ensure columns contain expected data
            airport_name = columns[0].text.strip()
            rating = columns[1].text.strip()
            airports.append({'Airport': airport_name, 'Rating': rating})
    
    return airports

def save_data_to_csv(data, output_path):
    """
    Saves the scraped data to a CSV file.

    Args:
        data (list): The scraped data.
        output_path (str): The file path to save the data.
    """
    if not data:
        print("No data to save.")
        return

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    df.to_csv(output_path, index=False)
    print(f"Data saved to '{output_path}'.")

if __name__ == "__main__":
    # Configuration
    URL = "https://skytraxratings.com/a-z-of-airport-ratings"
    OUTPUT_FILE = "data/airport_ratings.csv"  # Save in the 'data' folder within the project

    # Run data scraping
    print("Starting data scraping...")
    scraped_data = scrape_airport_ratings(URL)
    if scraped_data:
        save_data_to_csv(scraped_data, OUTPUT_FILE)
    else:
        print("No data was scraped.")
