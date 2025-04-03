from transformers import pipeline

# Define the model name (adjust if needed)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the pipeline, which will automatically download the model if not present
classifier = pipeline("sentiment-analysis", model=model_name)

print("Model and tokenizer downloaded successfully.")
