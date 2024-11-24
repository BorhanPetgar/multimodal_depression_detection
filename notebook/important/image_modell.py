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
import os

class ResNetBinaryClassifier:
    def __init__(self, data_dir, batch_size=32, learning_rate=0.001, num_epochs=5, train_val_split=0.8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.data_dir = data_dir
        self.train_val_split = train_val_split
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Initialize model, criterion, and optimizer
        self.model = self._initialize_model()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _create_data_loaders(self):
        # Data augmentation and normalization
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load dataset
        dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def _initialize_model(self):
        # Load pre-trained ResNet-18 model
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        model = model.to(self.device)
        return model

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for images, labels in tqdm(self.train_loader, desc='Training'):
            images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        correct_samples = []
        incorrect_samples = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                predictions = self.model(images)
                loss = self.criterion(predictions, labels)
                epoch_loss += loss.item()

                preds = torch.round(predictions)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                for i in range(len(labels)):
                    if preds[i] == labels[i]:
                        correct_samples.append((images[i].cpu(), labels[i].cpu()))
                    else:
                        incorrect_samples.append((images[i].cpu(), labels[i].cpu()))

        return (epoch_loss / len(self.val_loader), 
                np.array(all_preds).flatten(), 
                np.array(all_labels).flatten(),
                correct_samples, incorrect_samples)

    def fit(self):
        train_losses, val_losses = [], []

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_preds, val_labels, correct_samples, incorrect_samples = self.evaluate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')

        # Save correct and incorrect samples
        torch.save(correct_samples, 'correct_samples.pt')
        torch.save(incorrect_samples, 'incorrect_samples.pt')

        # Plot training and validation losses
        self._plot_losses(train_losses, val_losses)

        # Generate classification report and confusion matrix
        self._generate_report(val_labels, val_preds)

    def _plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_plot.png')
        plt.close()

    def _generate_report(self, labels, preds):
        print("Classification Report:\n", classification_report(labels, preds))
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')
        plt.close()

# Usage
if __name__ == "__main__":
    data_directory = '/home/fteam5/borhan/nlp/project/data/images'
    
    classifier = ResNetBinaryClassifier(
        data_dir=data_directory,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=5
    )
    classifier.fit()
