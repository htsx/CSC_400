import requests
from bs4 import BeautifulSoup
import csv

#URL for the A-Z airport review list
base_url = "https://www.airlinequality.com/review-pages/a-z-airport-reviews/"

#function to get the list of airport URLs with names
def get_airport_urls():
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Failed to fetch airport list page")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    airport_data = []

    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/airport-reviews/' in href:
            full_url = f"https://www.airlinequality.com{href}"
            airport_name = link.get_text(strip=True)
            airport_data.append((airport_name, full_url))

    print(f"Found {len(airport_data)} airport review pages")
    return airport_data

#function to scrape reviews from airport review page's
def scrape_reviews(airport_name, airport_url):
    response = requests.get(airport_url)
    if response.status_code != 200:
        print(f"Failed to fetch reviews for {airport_url}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []

    for container in soup.find_all('article', {'itemprop': 'review'}):
        review_text = container.find('div', {'class': 'text_content'}).get_text(strip=True)
        passenger_name = container.find('span', {'itemprop': 'author'}).get_text(strip=True) if container.find('span', {'itemprop': 'author'}) else 'Unknown'
        review_date = container.find('time')['datetime'] if container.find('time') else 'Unknown'

        reviews.append({
            'airport_name': airport_name,
            'passenger_name': passenger_name,
            'review_date': review_date,
            'review_text': review_text
        })

    print(f"Scraped {len(reviews)} reviews from {airport_url}")
    return reviews

#main
if __name__ == "__main__":
    airport_data = get_airport_urls()
    all_reviews = []

    #scrape reviews from the first 5 airport pages
    for airport_name, url in airport_data[:5]:
        reviews = scrape_reviews(airport_name, url)
        all_reviews.extend(reviews)

    #save raw reviews to csv file
    with open('data/raw_reviews.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['airport_name', 'passenger_name', 'review_date', 'review_text'])
        writer.writeheader()
        writer.writerows(all_reviews)

    print("Raw reviews saved to raw_reviews.csv")