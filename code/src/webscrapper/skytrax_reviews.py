import requests
from bs4 import BeautifulSoup
import csv
import time

#urls for the Skytrax A-Z airport and airline review lists
airport_base_url = "https://www.airlinequality.com/review-pages/a-z-airport-reviews/"
airline_base_url = "https://www.airlinequality.com/review-pages/a-z-airline-reviews/"

#function to get the list of review urls with names
def get_review_urls(base_url, review_type):
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to fetch {review_type} list page")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    review_data = []

    #find all review links available
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/airport-reviews/' in href or '/airline-reviews/' in href:  # Skytrax review URLs
            full_url = f"https://www.airlinequality.com{href}"
            review_name = link.get_text(strip=True)
            review_data.append((review_name, full_url, review_type))

    print(f"Found {len(review_data)} {review_type} review pages")
    return review_data

#scrape reviews from a review page
def scrape_reviews(review_name, review_url, review_type):
    response = requests.get(review_url)
    if response.status_code != 200:
        print(f"Failed to fetch reviews for {review_url}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []

    #all review containers
    for container in soup.find_all('article', {'itemprop': 'review'}):
        review_text = container.find('div', {'class': 'text_content'}).get_text(strip=True) if container.find('div', {'class': 'text_content'}) else 'Unknown'
        passenger_name = container.find('span', {'itemprop': 'author'}).get_text(strip=True) if container.find('span', {'itemprop': 'author'}) else 'Unknown'
        review_date = container.find('time')['datetime'] if container.find('time') else 'Unknown'

        reviews.append({
            'review_name': review_name,
            'review_type': review_type,
            'passenger_name': passenger_name,
            'review_date': review_date,
            'review_text': review_text
        })

    print(f"Scraped {len(reviews)} reviews from {review_url}")
    return reviews

#main
if __name__ == "__main__":
    #get airport and airline review urls for scrapping
    airport_data = get_review_urls(airport_base_url, 'airport')
    airline_data = get_review_urls(airline_base_url, 'airline')
    all_reviews = []

    #scrape reviews from every airport and airline page found
    for review_name, url, review_type in airport_data + airline_data:
        reviews = scrape_reviews(review_name, url, review_type)
        all_reviews.extend(reviews)
        time.sleep(1)  # Be kind to the server, avoid hammering it with requests

    #save raw reviews to a csv file
    with open('data/raw_skytrax_reviews.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['review_name', 'review_type', 'passenger_name', 'review_date', 'review_text'])
        writer.writeheader()
        writer.writerows(all_reviews)

    print("Raw reviews saved to raw_skytrax_reviews.csv")