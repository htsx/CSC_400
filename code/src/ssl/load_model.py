from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to your local model directory where the files are stored
model_path = "./model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
