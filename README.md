CSC 400 Research Project: Flight Experience Feedback Analysis

The project focuses on analyzing airline passenger reviews using sentiment analysis techniques to better understand customer satisfaction. The goal is to compare the effectiveness of three methods:

- Word-Score Sentiment Scoring

- Emotion Lexicon-based Sentiment Analysis

- Deep Learning-Based Sentiment Analysis

To evaluate these techniques, a hybrid labeled dataset is created using a hybrid labeling approach (rule-based + lexicon-based methods) followed by pseudo-labeling with . This dataset is not used for training, but solely for evaluating the performance of each technique. Evaluation is based on four key metrics: F1-score, Accuracy, Precision, and Recall.

The results, visualizations, and sample reviews are displayed through a Flask-based dashboard.

## 🔧 Setting Up the Virtual Environment (`venv`)

This project uses a Python virtual environment to manage dependencies.
Follow the steps below to create and activate the environment based on your operating system.

---

### 🪟 For Windows

1. **Open Command Prompt or PowerShell**
2. **Navigate to your project directory**  
   ```bash
   cd path\to\your\project
   ```
3. **Create the virtual environment**  
   ```bash
   python -m venv venv
   ```
4. **Activate the virtual environment**  
   ```bash
   venv\Scripts\activate
   ```

You should now see `(venv)` in your terminal prompt, indicating it's active.

---

### 🍎 For macOS / Linux

1. **Open Terminal**
2. **Navigate to your project directory**  
   ```bash
   cd /path/to/your/project
   ```
3. **Create the virtual environment**  
   ```bash
   python3 -m venv venv
   ```
4. **Activate the virtual environment**  
   ```bash
   source venv/bin/activate
   ```

You should now see `(venv)` in your terminal prompt, indicating it's active.

---

### 📦 Installing Dependencies

Once the virtual environment is activated, install all required packages using:

```bash
pip install -r requirements.txt
```

This will install everything your project needs to run.

---

### 🧪 Verify the Environment

To check that packages are installed:

```bash
pip list
```

You should see a list of installed dependencies, including those from `requirements.txt`.

---

### ❌ Deactivate the Environment

To exit the virtual environment:

```bash
deactivate
```
---
**Libraries Used**

This project leverages several Python libraries to analyze airline passenger reviews using sentiment analysis techniques. Below is a list of the key libraries used in this project and their specific purposes:

1. beautifulsoup4==4.13.4
Purpose: Used for web scraping to collect airline passenger reviews from websites. BeautifulSoup helps extract structured data from HTML and XML documents, which is essential for gathering customer feedback for sentiment analysis.

2. matplotlib==3.10.1
Purpose: A powerful library for creating static, animated, and interactive visualizations. Matplotlib is used in this project to generate various plots (e.g., bar charts, histograms, and pie charts) to visualize sentiment distributions and comparison results across different sentiment analysis methods.

3. seaborn==0.13.2
Purpose: Built on top of Matplotlib, Seaborn simplifies the process of creating attractive and informative statistical graphics. It's used to produce advanced visualizations like heatmaps, violin plots, and regression plots to explore relationships in the sentiment analysis results.

4. python-dotenv==1.1.0
Purpose: This library is used to securely manage configuration settings and sensitive data, such as API keys, by loading them from a .env file into environment variables. This is important for protecting sensitive information when interacting with external services, such as OpenAI’s API.

5. openai==0.28.0
Purpose: The official Python client for OpenAI’s models, used to interact with GPT and other advanced NLP models for sentiment analysis. In this project, OpenAI's models are used for deep learning-based sentiment analysis techniques, providing accurate and scalable predictions.

6. pandas==2.2.3
Purpose: Pandas is essential for data manipulation and analysis. It is used to preprocess and clean the review data, load it into data frames, and perform analysis. This library simplifies tasks like filtering, aggregating, and summarizing data from the dataset.

7. scikit_learn==1.6.1
Purpose: A widely used library for machine learning. Scikit-learn is used in this project for traditional machine learning methods like classification (e.g., Logistic Regression, Random Forest) to analyze sentiment in reviews. It also provides tools to evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

8. requests==2.32.3
Purpose: This library is used to send HTTP requests to APIs and download data from external sources. It's particularly useful in this project for fetching additional data, like reviews from online platforms or querying external services to enrich the dataset.

9. textblob==0.19.0
Purpose: TextBlob is a simple NLP library used to perform sentiment analysis on text. It provides basic polarity and subjectivity scores, which are used to classify reviews as positive, negative, or neutral, making it a valuable tool for scoring sentiment in this project.

10. transformers==4.51.2
Purpose: This library from Hugging Face provides access to state-of-the-art pre-trained models, such as BERT, RoBERTa, and GPT. In this project, it's used for deep learning-based sentiment analysis, allowing for more accurate and sophisticated sentiment classification by fine-tuning pre-trained models on the review dataset.

11. torch==2.1.0
Purpose: PyTorch is a deep learning framework that allows you to build and train custom neural networks. It is used in this project to implement and train sentiment analysis models, particularly for the deep learning-based approach. PyTorch also supports GPU acceleration, making it efficient for training large models.

