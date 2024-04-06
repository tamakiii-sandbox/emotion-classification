import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess the Emotion Dataset
dataset = load_dataset("emotion", split="train", trust_remote_code=True)
dataset_list = dataset.to_list()
train_data, test_data = train_test_split(dataset_list, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Tokenize and encode the text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(data):
    return tokenizer.batch_encode_plus(
        [item['text'] for item in data],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

train_encodings = tokenize_data(train_data)
val_encodings = tokenize_data(val_data)
test_encodings = tokenize_data(test_data)

train_labels = torch.tensor([item['label'] for item in train_data])
val_labels = torch.tensor([item['label'] for item in val_data])
test_labels = torch.tensor([item['label'] for item in test_data])

# Create PyTorch datasets and data loaders
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Fine-tune the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    model.eval()
    with torch.no_grad():
        val_accuracy = 0
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            val_accuracy += torch.sum(predictions == labels).item()
        
        val_accuracy /= len(val_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {val_accuracy:.4f}")

# Evaluate the model
# ... Make predictions on the test set and calculate evaluation metrics

# Analyze and visualize the results
# ... Create a confusion matrix and examine sample predictions

# Document and share your findings
# ... Write a README file and share your code on GitHub
