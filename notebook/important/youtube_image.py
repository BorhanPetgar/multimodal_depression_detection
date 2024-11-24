import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = datasets.ImageFolder(root='/home/fteam5/borhan/nlp/project/data/images', transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid()
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training and evaluation functions
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    model.train()
    
    for images, labels in tqdm(iterator):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    model.eval()
    all_preds = []
    all_labels = []
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            predictions = model(images)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            preds = torch.round(predictions)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for i in range(len(labels)):
                if preds[i] == labels[i]:
                    correct_samples.append((images[i].cpu(), labels[i].cpu()))
                else:
                    incorrect_samples.append((images[i].cpu(), labels[i].cpu()))
    
    return epoch_loss / len(iterator), all_preds, all_labels, correct_samples, incorrect_samples

# Training loop
N_EPOCHS = 5
train_losses, val_losses = [], []

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_preds, val_labels, correct_samples, incorrect_samples = evaluate(model, val_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {val_loss:.3f}')

# Save correct and incorrect samples
torch.save(correct_samples, 'correct_samples.pt')
torch.save(incorrect_samples, 'incorrect_samples.pt')

# Plotting and saving the loss
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.savefig('loss_plot.png')

# Classification report and confusion matrix
print(classification_report(val_labels, val_preds))
cm = confusion_matrix(val_labels, val_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')