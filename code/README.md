 Follow these instructions to set up the virtual environment and install the necessary dependencies for this project.

Prerequisites
Make sure you have the following installed:

Python 3.6 or higher
pip (Python package manager)
## Step 1: Create a Virtual Environment
Open a terminal or command prompt.

Navigate to the project folder where the reqs.txt file is located.

Run the following command to create a virtual environment: python3 -m venv venv

Activate the virtual environment: #On macOS/Linux: source venv/bin/activate

On Windows: venv\Scripts\activate

You’ll know the virtual environment is active when you see (venv) in your terminal prompt.

## Step 2: Install Dependencies
Once the virtual environment is active, run the following command to install the required dependencies: pip install -r requirements.txt

This will install the following Python libraries: pandas numpy matplotlib seaborn scikit-learn nltk beautifulsoup4 vaderSentiment textblob flask (for the interactive dashboard) dash (for dashboard creation)

## Step 3: Verify Installation
To ensure all dependencies are installed correctly, you can run: pip list

You should see all the listed libraries in the output.

## Step 4: Deactivate the Virtual Environment (If needed)
When you’re done working, deactivate the virtual environment by running: deactivate
