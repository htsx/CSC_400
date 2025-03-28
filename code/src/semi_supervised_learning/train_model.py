from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Load the data
data = pd.read_csv("C:/Users/twoce/OneDrive/CSC 400/csc_400/code/src/semi_supervised_learning/data/processed_reviews.csv")

# Filter for labeled data (first 1000)
labeled_data = data.iloc[:1000]

# Ensure labels are not NaN and are converted to strings
labeled_data['ground_truth_sentiment'] = labeled_data['ground_truth_sentiment'].fillna('neutral').astype(str)

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Add the 'NEUTRAL' label to the label_map
        self.label_map = {'positive': 1, 'negative': 0, 'neutral': 2}
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
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
        
        # Convert the label to a tensor with numeric values
        label = str(self.labels[idx]).lower()  # Convert to string and lowercase for consistency
        if label not in self.label_map:
            label = 'neutral'  # Default to neutral if the label is invalid
        label = torch.tensor(self.label_map[label])  # Map label to numeric value
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}


# Initialize the tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Now you can create the dataset with the tokenizer
train_dataset = ReviewDataset(
    labeled_data['review_text'].to_numpy(), 
    labeled_data['ground_truth_sentiment'].to_numpy(), 
    tokenizer
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load BERT for sequence classification with 3 labels
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 labels for positive, negative, neutral

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./fine_tuned_model')

print("Model training complete and saved!")
