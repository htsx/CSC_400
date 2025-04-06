CSC 400 Research Project: Flight Experience Feedback Analysis

Your project focuses on analyzing airline passenger reviews using sentiment analysis techniques to better understand customer satisfaction. The goal is to compare the effectiveness of three methods:

- Sentiment Scoring & Distribution Analysis

- Keyword & Topic-Based Sentiment Analysis

- Deep Learning-Based Sentiment Analysis

To evaluate these techniques, a semi-supervised labeled dataset is created using a hybrid labeling approach (rule-based + lexicon-based methods) followed by pseudo-labeling. This dataset is not used for training, but solely for evaluating the performance of each technique. Evaluation is based on four key metrics: F1-score, Accuracy, Precision, and Recall.

The results, visualizations, and sample reviews are displayed through a Flask-based dashboard.

Absolutely! Here’s the updated `README.md` section with steps on how to install dependencies using `requirements.txt`:

```markdown
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
