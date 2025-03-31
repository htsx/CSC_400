# CSC_400
Flight Experience Feedback Sentiment Analysis Project

This project aims to:

- Compare three data analysis techniques for sentiment analysis to determine which one provides the most accurate insights into passenger feedback.
- Analyze passenger reviews to assess how different factors like service quality, comfort, and punctuality affect satisfaction levels.
- Provide data-driven conclusions on the most effective techniques for sentiment analysis in passenger feedback analysis.

Using these 3 Data Analysis Techniques:

- Sentiment Scoring & Distribution Analysis
- Keyword & Topic Based Sentiment Analysis
- Deep Learning-Based Sentiment Analysis

You must setup the Virtual Environment for this to work, guide below.

## Virtual Environment Guide
__________________________________________________________________________________________________________________________________
*****Prerequisites*****
Make sure you have the following installed:
- Python 3.6 or higher to 3.11
- `pip` (Python package manager)
__________________________________________________________________________________________________________________________________
*****Creating the Virtual Environment*****
1. Open the Virtual Studio Code (vscode) application.
2. Clone the 'CSC_400' github repository (this repository) if you haven't already to your system using vscode.
3. Open the terminal or command prompt in vscode.
4. Navigate to the code folder in this project (`cd CSC_400` > `cd code` in terminal/command prompt).
5. Run the following command to create a virtual environment: `python3 -m venv venv`.
__________________________________________________________________________________________________________________________________
*****Activating the Virtual Environment*****
1. After creating the virtual environment, you must activate the virtual environment ny doing the following commands: <br />
   macOS/Linux do: `source venv/bin/activate`<br />
   On Windows do: `.\venv\Scripts\activate`<br />
   (You’ll know the virtual environment is active when you see `(venv)` in your terminal prompt.).<br />
2. Install the required Python libraries by running: `pip install -r requirements.txt` while the virtual environment is running.
3. Run the Flask application:: `py dashboard/app.py` or `python dashboard/app.py`.
4. When you’re done working, deactivate the virtual environment by running: `deactivate`.
__________________________________________________________________________________________________________________________________
*****Verify Installation*****
1. To ensure all dependencies are installed correctly, you can run: `pip list`.
2. You should see all the listed libraries in the output.
__________________________________________________________________________________________________________________________________
*****The following Python libraries will be installed from requirements.txt:*****
- beautifulsoup4 - For extracting data from web pages by parsing HTML or XML content (web scraping).
- pandas - For reading, processing, and manipulating datasets (e.g., CSV files, reviews data).
- textblob - For basic text analysis tasks like sentiment detection, spell checking, and text correction (used for sentiment scoring and distribution analysis).
- vaderSentiment - For analyzing text sentiment (positive, negative, or neutral scores) (alternative to TextBlob, often used for short and social media texts).
- matplotlib - For creating static visualizations like line graphs, bar plots, and scatter plots (visualizing sentiment analysis results).
- flask - For creating a lightweight web application or API backend (serving the sentiment analysis results on a dashboard).
- flask-WTF - For creating and validating web forms with CSRF protection and integration with Flask (useful if handling user-submitted data or reviews).
- flask-SQLAlchemy - For integrating SQLAlchemy with Flask to manage database interactions easily (useful if storing reviews and sentiment scores in a database).
- nltk - Provides tools for natural language processing, such as tokenization, stopword removal, and part-of-speech tagging (used for pre-processing text before analysis).
- gensim - Implements topic modeling and word embeddings for analyzing larger datasets (used in keyword and topic-based sentiment analysis).
- transformers - Provides pre-trained deep learning models (e.g., BERT, GPT) for advanced sentiment analysis and text classification (used in deep learning-based sentiment analysis).
- torch - The core deep learning framework used by models in the transformers library (required for deep learning-based sentiment analysis).
- scikit-learn - A comprehensive library for machine learning (used for feature extraction, model training, and evaluation in sentiment analysis).
- datasets - A library for easily accessing and sharing datasets, especially in the field of machine learning (used for obtaining review datasets for analysis).
- accelerate - A library for simplifying training and evaluation of models on multiple hardware devices (used for optimizing deep learning workflows in sentiment analysis).
- seaborn - A statistical data visualization library based on matplotlib (used for creating more advanced plots like heatmaps and correlation matrices).
