## *****Prerequisites*****

Make sure you have the following installed:
- Python 3.6 or higher
- `pip` (Python package manager)

## Step 1: Create a Virtual Environment

1. Open a terminal or command prompt.
2. Run the following command to create a virtual environment:
python3 -m venv venv
3. Navigate to the project folder where the `requirements.txt` file is located.
4. pip install -r requirements.txt
5. Activate the virtual environment:
On macOS/Linux:
source venv/bin/activate

On Windows:
.\venv\Scripts\activate

You’ll know the virtual environment is active when you see `(venv)` in your terminal prompt.

## Step 2: Install Dependencies
1. Once the virtual environment is active, copy and past this list of dependencies into requirements.txt:

## Dependencies
beautifulsoup4
pandas
vaderSentiment
matplotlib
flask
flask-WTF
flask-SQLAlchemy

This will install the following Python libraries:
- beautifulsoup4 - For extracting data from web pages by parsing HTML or XML content (web scraping).
- pandas - For reading, processing, and manipulating datasets (e.g., CSV files, reviews data).
- textblob - For basic text analysis tasks like sentiment detection, spell checking, and text correction.
- vaderSentiment - For analyzing text sentiment (positive, negative, or neutral scores) (optional, choose between TextBlob and VADER).
- matplotlib - For creating static visualizations like line graphs, bar plots, and scatter plots (visualizing sentiment analysis results).
- flask - For creating a lightweight web application or API backend (serving the scraped data and sentiment analysis results on a website).
- flask-WTF - For creating and validating web forms with CSRF protection and integration with Flask (useful if handling user-submitted data or reviews).
- flask-SQLAlchemy - For integrating SQLAlchemy with Flask to manage database interactions easily (useful if storing reviews and sentiment scores in a database).


## Step 3: Verify Installation
1. To ensure all dependencies are installed correctly, you can run:
pip list

2. You should see all the listed libraries in the output.

## Step 4: Deactivate the Virtual Environment (Optional)
When you’re done working, deactivate the virtual environment by running:
deactivate

Now your project environment is ready to use! If you encounter any issues, make sure the required Python version is installed and that you’ve activated the virtual environment correctly.
