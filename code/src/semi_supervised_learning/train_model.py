from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np

#Define a proper Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
        
    def __len__(self):
        return len(self.labels)

#Load and prepare data
df = pd.read_csv("data/processed_reviews.csv")
texts = df['review_text'].astype(str).values[:1000]  # Ensure text is string
labels = df['ground_truth_sentiment'].values[:1000]

#Clean and map labels
label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
def clean_label(label):
    if pd.isna(label):
        return 1  # Default to neutral
    return label_map.get(str(label).lower().strip(), 1)  # Default to neutral if unknown

labels = torch.tensor([clean_label(l) for l in labels])

#Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

#Initialize model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Tokenize data
def tokenize(texts):
    return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=128)

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

#Create proper datasets
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

#Training configuration
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    eval_strategy="epoch",  #Updated from evaluation_strategy
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

#Initialize and run trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting training...")
trainer.train()
print(f"Training completed in {trainer.state.log_history[-1]['train_runtime']:.2f} seconds")

#Save model
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')