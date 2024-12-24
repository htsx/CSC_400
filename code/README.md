Follow these instructions to set up the virtual environment and install the necessary dependencies for this project.
Prerequisites
Make sure you have the following installed:

Python 3.6 or higher
pip (Python package manager)
## Step 1: Create a Virtual Environment

Create the Virtual Environment: Open a terminal or command prompt and run the following command (you can choose any directory to create the virtual environment):

On macOS/Linux:
python3 -m venv ~/myenv

On Windows:
python -m venv C:\path\to\myenv
This creates a virtual environment in the specified location (e.g., ~/myenv on macOS/Linux or C:\path\to\myenv on Windows).

## Step 2: Activate the Virtual Environment
On macOS/Linux:
bash
Copy code
source ~/myenv/bin/activate
On Windows:
bash
Copy code
C:\path\to\myenv\Scripts\activate
You’ll know the virtual environment is active when you see (myenv) in your terminal prompt.

## Step 3: Install Dependencies
Once the virtual environment is active, navigate to your project folder (where the requirements.txt file is located) and run the following command to install the required dependencies:

bash
Copy code
pip install -r requirements.txt
This will install the following Python libraries listed in the requirements.txt file:

pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
beautifulsoup4
vaderSentiment
textblob
flask (for the interactive dashboard)
dash (for dashboard creation)
## Step 4: Verify Installation
To ensure all dependencies are installed correctly, you can run:

bash
Copy code
pip list
You should see all the listed libraries in the output.

## Step 5: Deactivate the Virtual Environment (If needed)
When you’re done working, deactivate the virtual environment by running:

bash
Copy code
deactivate
