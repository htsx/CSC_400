import pandas as pd
import matplotlib.pyplot as plt

# Load sentiment analysis results
df = pd.read_csv("data/sd_results.csv")

# Count occurrences of each sentiment
vader_counts = df["VADER_Sentiment"].value_counts()
textblob_counts = df["TextBlob_Sentiment"].value_counts()

# Plot VADER Sentiment Analysis
plt.figure(figsize=(6, 4))
vader_counts.plot(kind="bar", color=["red", "blue", "green"])
plt.title("VADER Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Plot TextBlob Sentiment Analysis
plt.figure(figsize=(6, 4))
textblob_counts.plot(kind="bar", color=["red", "blue", "green"])
plt.title("TextBlob Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

