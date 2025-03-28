import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the unlabeled reviews (this will be your 10000 new reviews that need predictions)
reviews = pd.read_csv("C:/Users/twoce/OneDrive/CSC 400/csc_400/code/data/ground_truth/ground_truth_reviews.csv")
reviews = reviews.iloc[1000:]  # Skip the first 1000 since those are already labeled manually

class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length=128):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews.iloc[idx])  # Convert review to string
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Create dataset and dataloader
unlabeled_dataset = ReviewDataset(reviews['review_text'], tokenizer)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False)

# Function to get predictions and confidence scores
def predict(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    confidences = []
    
    with torch.no_grad():  # No need to compute gradients during prediction
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1)  # Remove extra batch dimension
            attention_mask = batch['attention_mask'].squeeze(1)
            
            # Get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get the class prediction (index of max probability)
            predicted_class = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1).values  # Confidence is the max probability
            
            # Append predictions and confidence scores
            predictions.extend(predicted_class.tolist())
            confidences.extend(confidence.tolist())
    
    return predictions, confidences

# Get predictions and confidence scores
predictions, confidences = predict(model, unlabeled_loader)

# Add predictions and confidence scores to the DataFrame
reviews['predicted_sentiment'] = predictions
reviews['confidence'] = [round(conf, 2) for conf in confidences]  # Round confidence to 2 decimal places

# Map numeric predictions back to labels (you can adjust these as per your label mapping)
label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
reviews['predicted_sentiment'] = reviews['predicted_sentiment'].map(label_map)

# Save the predictions, confidence, and all original columns to a CSV file
reviews.to_csv('data/model_predictions.csv', index=False)

print("Predictions with confidence scores saved to model_predictions.csv")
