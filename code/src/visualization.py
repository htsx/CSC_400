import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sentiment analysis results
def load_data():
    return pd.read_csv('data/sentiment_reviews.csv')

# Function to plot sentiment distribution
def plot_sentiment_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['sentiment_score'], kde=True, color='skyblue', bins=30)
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Function to display top 10 airports by sentiment score
def display_top_airports(data):
    # Sorting by sentiment score
    top_airports = data[['Airport', 'sentiment_score']].sort_values(by='sentiment_score', ascending=False).head(10)
    st.write("Top 10 Airports by Sentiment Score")
    st.write(top_airports)

# Interactive options
def display_dashboard():
    st.title("Airport Ratings Sentiment Dashboard")
    
    # Load the data
    data = load_data()

    # Displaying a sample of the data
    st.write("Sample of the Sentiment Analysis Data:")
    st.write(data.head())

    # Plot sentiment distribution
    plot_sentiment_distribution(data)

    # Display top 10 airports
    display_top_airports(data)

if __name__ == "__main__":
    display_dashboard()
