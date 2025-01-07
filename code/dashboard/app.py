from dash import Dash, dcc, html
import pandas as pd
import plotly.express as px

app = Dash(__name__)

# Load data
data = pd.read_csv('../data/sentiment_reviews.csv')

# Layout
app.layout = html.Div([
    html.H1("Airline Customer Feedback Dashboard"),
    dcc.Graph(figure=px.histogram(data, x='sentiment_score', title='Sentiment Distribution'))
])

if __name__ == '__main__':
    app.run_server(debug=True)
