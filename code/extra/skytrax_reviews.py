import requests
from bs4 import BeautifulSoup
import csv
import time
import random

#URLs for the Skytrax A-Z airport and airline review lists
airport_base_url = "https://www.airlinequality.com/review-pages/a-z-airport-reviews/"
airline_base_url = "https://www.airlinequality.com/review-pages/a-z-airline-reviews/"

#Function to get the list of review URLs with names
def get_review_urls(base_url, review_type, session):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    retries = 3
    for attempt in range(retries):
        try:
            response = session.get(base_url, headers=headers)
            response.raise_for_status()  #If there's a bad response, raise an error
            soup = BeautifulSoup(response.content, 'html.parser')
            review_data = []

            #Loop through all the links to find reviews
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/airport-reviews/' in href or '/airline-reviews/' in href:  #Skytrax review URLs
                    full_url = f"https://www.airlinequality.com{href}"
                    review_name = link.get_text(strip=True)
                    review_data.append((review_name, full_url, review_type))

            print(f"Found {len(review_data)} {review_type} review pages")
            return review_data
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {review_type} list page (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(random.randint(3, 7))  #Wait for a bit before trying again
            else:
                print("Max retries reached. Giving up.")
                return []

#Function to classify review based on rating value
def classify_review(rating_value):
    if rating_value >= 7:
        return 'Positive'
    elif 4 <= rating_value < 7:
        return 'Neutral'
    else:
        return 'Negative'

#Function to scrape reviews from a given page
def scrape_reviews(review_name, review_url, review_type, session):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    retries = 3
    for attempt in range(retries):
        try:
            response = session.get(review_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            reviews = []

            #Extract reviews from the page
            for container in soup.find_all('article', {'itemprop': 'review'}):
                review_text = container.find('div', {'class': 'text_content'}).get_text(strip=True) if container.find('div', {'class': 'text_content'}) else 'Unknown'
                passenger_name = container.find('span', {'itemprop': 'author'}).get_text(strip=True) if container.find('span', {'itemprop': 'author'}) else 'Unknown'
                review_date = container.find('time')['datetime'] if container.find('time') else 'Unknown'

                rating_value = float(container.find('span', {'itemprop': 'ratingValue'}).get_text(strip=True)) if container.find('span', {'itemprop': 'ratingValue'}) else 0.0
                
                #Placeholder for review classification in case it's missing
                review_classification = "NotClassified"  #Just putting something here to avoid leaving it empty

                reviews.append({
                    'review_name': review_name,
                    'review_type': review_type,
                    'passenger_name': passenger_name,
                    'review_date': review_date,
                    'review_text': review_text,
                    'rating_value': rating_value,
                    'review_classification': review_classification
                })

            print(f"Scraped {len(reviews)} reviews from {review_url}")
            return reviews
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch reviews for {review_url} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(random.randint(3, 7))  #Wait a little and try again
            else:
                print("Max retries reached. Abandoning this page.")
                return []

#Main function to get review URLs and scrape data
if __name__ == "__main__":
    session = requests.Session()  #Reuse the session across requests to save time

    #Get review URLs for airports and airlines
    airport_data = get_review_urls(airport_base_url, 'airport', session)
    airline_data = get_review_urls(airline_base_url, 'airline', session)
    all_reviews = []

    #Scrape reviews from each page found
    for review_name, url, review_type in airport_data + airline_data:
        reviews = scrape_reviews(review_name, url, review_type, session)
        if reviews:
            all_reviews.extend(reviews)
        time.sleep(1)  #Optional: Give a small break between pages to avoid getting blocked

    #Save all reviews to a CSV file
    with open('../data/extra/raw_skytrax_reviews.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['review_name', 'review_type', 'passenger_name', 'review_date', 'review_text', 'rating_value', 'review_classification'])
        writer.writeheader()
        writer.writerows(all_reviews)

    print("Raw reviews saved to raw_skytrax_reviews.csv")