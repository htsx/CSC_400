## *****Prerequisites*****

Make sure you have the following installed:
- Python 3.6 or higher
- `pip` (Python package manager)

## Step 1: Create a Virtual Environment

1. Open a terminal or command prompt.
2. Navigate to the project folder where the `requirements.txt` file is located.
3. Run the following command to create a virtual environment:
python3 -m venv venv

4. Activate the virtual environment:
On macOS/Linux:
source venv/bin/activate

On Windows:
venv\Scripts\activate

You’ll know the virtual environment is active when you see `(venv)` in your terminal prompt.

## Step 2: Install Dependencies
1. Once the virtual environment is active, copy and past this list of dependencies into requirements.txt:

pandas               # For reading, processing, and manipulating datasets (e.g., CSV files).  
numpy                # For performing numerical operations and handling arrays.  
matplotlib           # For creating static visualizations like line graphs, bar plots, and scatter plots.  
seaborn              # For generating advanced and aesthetically pleasing statistical visualizations.  
scikit-learn         # For machine learning tasks, such as text classification and dataset splitting.  
nltk                 # For working with text data, such as tokenization, stemming, and stopword removal.  
beautifulsoup4       # For extracting data from web pages by parsing HTML or XML content.  
vaderSentiment       # For analyzing text sentiment (positive, negative, or neutral scores).  
textblob             # For basic text analysis tasks like sentiment detection, spell checking, and text correction.  
flask                # For creating a lightweight web application or API backend.
flask-WTF
flask-SQLAlchemy
dash                 # For building interactive, web-based dashboards to visualize and explore data.  

After that, run the following command to install the required dependencies:
pip install -r requirements.txt

This will install the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- beautifulsoup4
- vaderSentiment
- textblob
- flask (for the interactive dashboard)
- dash (for dashboard creation)

## Step 3: Verify Installation
1. To ensure all dependencies are installed correctly, you can run:
pip list

2. You should see all the listed libraries in the output.

## Step 4: Deactivate the Virtual Environment (Optional)
When you’re done working, deactivate the virtual environment by running:
deactivate

Now your project environment is ready to use! If you encounter any issues, make sure the required Python version is installed and that you’ve activated the virtual environment correctly.
