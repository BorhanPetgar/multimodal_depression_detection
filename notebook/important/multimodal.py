import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm
from PIL import Image
import os

# Download NLTK data
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# Load GloVe embeddings
def load_glove_embeddings(filename):
    embeddings = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings('glove.6B.50d.txt')

# Data Preprocessing
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word in glove_embeddings]
    return np.array([glove_embeddings[word] for word in tokens if word in glove_embeddings], dtype=float)

def pad_sequence(sequence, max_len=70, embed_size=50):
    if len(sequence) > max_len:
        return sequence[:max_len]
    else:
        pad = np.zeros((max_len - len(sequence), embed_size))
        return np.vstack([sequence, pad])

# Custom Dataset
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Preprocess text
        text_vectors = preprocess_text(text)
        text_vectors = pad_sequence(text_vectors)
        
        return image, torch.tensor(text_vectors, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Transform for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = MultimodalDataset('/content/merged_shuffled.csv', transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the Image Model (ResNet18)
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 1), nn.Sigmoid())
    
    def forward(self, x):
        return self.resnet(x)

# Define the Text Model (LSTM)
class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64 * 70, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), -1)
        return self.sigmoid(self.fc(x))

# Instantiate models, loss, and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_model = ImageModel().to(device)
text_model = TextModel().to(device)

criterion = nn.BCELoss()
optimizer_img = optim.Adam(image_model.parameters(), lr=0.001)
optimizer_txt = optim.Adam(text_model.parameters(), lr=0.001)

# Training Loop
def train_model(model, loader, optimizer):
    model.train()
    total_loss = 0
    for images, texts, labels in loader:
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images if model == image_model else texts)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation
def evaluate_model(image_model, text_model, loader):
    image_model.eval()
    text_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, texts, labels in loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            img_preds = image_model(images).squeeze()
            txt_preds = text_model(texts).squeeze()
            
            combined_preds = (img_preds > 0.5).int() + (txt_preds > 0.5).int()
            majority_vote = (combined_preds >= 1).int()
            
            all_preds.extend(majority_vote.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return classification_report(all_labels, all_preds)

# Train and Evaluate
for epoch in range(5):
    train_model(image_model, train_loader, optimizer_img)
    train_model(text_model, train_loader, optimizer_txt)
    print(f"Epoch {epoch+1} completed")

print("Validation Results:")
print(evaluate_model(image_model, text_model, val_loader))

print("Test Results:")
print(evaluate_model(image_model, text_model, test_loader))
