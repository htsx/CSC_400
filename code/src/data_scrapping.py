import csv

#function for scraping airport ratings
def scrape_airport_ratings(url):
    print("Web scraping functionality will be implemented later.")
    #data
    airports = [
        ["Abuja Nnamdi Azikwe International Airport", "2"],
        ["Accra Kotoka International Airport", "3"],
        ["Addis Ababa Bole International Airport", "2"]
    ]
    return airports

#function to save data to CSV
def save_data_to_csv(data, filename):
    if not data:
        print("No data to save.")
        return

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Airport', 'Rating'])  #header
        writer.writerows(data)  #data rows
    print(f"Data saved to '{filename}'.")

#main
if __name__ == "__main__":
    url = "https://skytraxratings.com/a-z-of-airport-ratings"
    output_file = "data/airport_ratings.csv"

    print("Starting data scraping...")
    data = scrape_airport_ratings(url)
    if data:
        save_data_to_csv(data, output_file)
    else:
        print("No data was scraped.")