import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Load the sentiment data
data = pd.read_csv('../data/sentiment_reviews.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Create a figure for sentiment distribution
fig_sentiment = px.histogram(data, x="sentiment_score", nbins=30, title="Sentiment Score Distribution", 
                              labels={"sentiment_score": "Sentiment Score"})

# Create a figure for average sentiment by airport
if 'Airport' in data.columns:
    avg_sentiment = data.groupby('Airport')['sentiment_score'].mean().reset_index()
    fig_airport_sentiment = px.bar(avg_sentiment, x="Airport", y="sentiment_score", 
                                   title="Average Sentiment Score by Airport",
                                   labels={"sentiment_score": "Average Sentiment Score"})
else:
    fig_airport_sentiment = {}

# Layout of the app
app.layout = html.Div([
    html.H1("Sentiment Analysis Dashboard"),
    dcc.Graph(figure=fig_sentiment),  # Display the sentiment score distribution
    html.H2("Sentiment by Airport"),
    dcc.Graph(figure=fig_airport_sentiment)  # Display sentiment by airport (if available)
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
