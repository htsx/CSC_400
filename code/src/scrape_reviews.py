import requests
from bs4 import BeautifulSoup

# Initialize an empty list to store the airline names
airline_names = []

# Updated URL for airport reviews
URL = "https://www.airlinequality.com/review-pages/a-z-airport-reviews/"

# Headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# Function to scrape reviews
def scrape_reviews(url, num_pages=1):
    reviews = []
    for page in range(1, num_pages + 1):
        response = requests.get(f"{url}page/{page}/", headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # Updated selector for review text - adjust as needed
        comments = soup.find_all('div', class_='text_content')
        for comment in comments:
            text = comment.get_text(strip=True)
            reviews.append(text)

    return reviews

if __name__ == "__main__":
    reviews = scrape_reviews(URL, num_pages=2)
    if reviews:
        for review in reviews[:5]:
            print(review)
    else:
        print("No reviews found.")