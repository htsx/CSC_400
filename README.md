# CSC_400
Flight Experience Feedback Analysis Project

This project aims to:
  1.  Identify and categorize themes in passenger feedback to highlight common concerns and preferences.
  2.  Quantify how various aspects, such as staff behavior, comfort, and punctuality, influence overall satisfaction.
  3.  Deliver data-driven recommendations for targeted service enhancements.

## Virtual Environment Guide

## *****Prerequisites*****

Make sure you have the following installed:
- Python 3.6 or higher
- `pip` (Python package manager)

## Create the Virtual Environment

1. Open the Virtual Studio Code (vscode) application.
2. Clone the 'CSC_400' github repository (this repository) if you haven't already to your system using vscode.
3. Open the terminal or command prompt in vscode.
4. Navigate to the code folder in this project (`cd CSC_400` > `cd code` in terminal/command prompt).
5. Run the following command to create a virtual environment: `python3 -m venv venv`.
6. Run the following command: `pip install -r requirements.txt`.
7. Activate the virtual environment, On macOS/Linux do: `source venv/bin/activate`, On Windows do: `.\venv\Scripts\activate` (You’ll know the virtual environment is active when you see `(venv)` in your terminal prompt.).
8. After activating the virtual environment, run this command to run the flask application: `py dashboard/app.py` or `python dashboard/app.py`.
9. When you’re done working, deactivate the virtual environment by running: `deactivate`.

## The following Python libraries will be installed from requirements.txt:
- beautifulsoup4 - For extracting data from web pages by parsing HTML or XML content (web scraping).
- pandas - For reading, processing, and manipulating datasets (e.g., CSV files, reviews data).
- textblob - For basic text analysis tasks like sentiment detection, spell checking, and text correction.
- vaderSentiment - For analyzing text sentiment (positive, negative, or neutral scores) (optional, choose between TextBlob and VADER).
- matplotlib - For creating static visualizations like line graphs, bar plots, and scatter plots (visualizing sentiment analysis results).
- flask - For creating a lightweight web application or API backend (serving the scraped data and sentiment analysis results on a website).
- flask-WTF - For creating and validating web forms with CSRF protection and integration with Flask (useful if handling user-submitted data or reviews).
- flask-SQLAlchemy - For integrating SQLAlchemy with Flask to manage database interactions easily (useful if storing reviews and sentiment scores in a database).

## Verify Installation
1. To ensure all dependencies are installed correctly, you can run: `pip list`.
2. You should see all the listed libraries in the output.
