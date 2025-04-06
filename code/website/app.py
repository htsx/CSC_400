from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/dashboard")
def dashboard():
    return redirect(url_for('all_techniques'))  # Redirecting to the all_techniques page

@app.route("/all_techniques")
def all_techniques():
    return render_template("all_techniques.html")

@app.route("/scoring_distribution_analysis")
def scoring_distribution_analysis():
    return render_template("scoring_distribution_analysis.html")

@app.route("/keywords_topics_analysis")
def keywords_topics_analysis():
    return render_template("keywords_topics_analysis.html")

@app.route("/deep_learning_based_analysis")
def deep_learning_based_analysis():
    return render_template("deep_learning_based_analysis.html")

if __name__ == '__main__':
    app.run(debug=True)
