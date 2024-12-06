import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np

# Define the Learnable Majority Voting Meta-Model
class LearnableMajorityVoting(nn.Module):
    def __init__(self):
        super(LearnableMajorityVoting, self).__init__()
        # Initialize learnable weights for text and image probabilities
        self.w_text = nn.Parameter(torch.tensor(np.random.uniform(0.4, 0.6), dtype=torch.float32))  # Weight for text model
        self.w_image = nn.Parameter(torch.tensor(np.random.uniform(0.4, 0.6), dtype=torch.float32))  # Weight for image model

    def forward(self, text_prob, image_prob):
        # Apply softmax to ensure weights are between 0 and 1 and sum to 1
        weights = torch.softmax(torch.stack([self.w_text, self.w_image]), dim=0)
        
        # Compute weighted sum of probabilities
        combined = weights[0] * text_prob + weights[1] * image_prob
        return torch.sigmoid(combined)  # Apply sigmoid to get the final probability

# Load CSV data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    features = df[['text_prob', 'image_prob']].values
    labels = df['label'].values
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# Accuracy computation
def compute_accuracy(meta_model, features, labels):
    meta_model.eval()
    with torch.no_grad():
        text_prob, image_prob = features[:, 0], features[:, 1]
        outputs = meta_model(text_prob, image_prob)
        preds = (outputs > 0.5).int()
        return accuracy_score(labels.cpu(), preds.cpu())

# Train Function
def train_meta_model(meta_model, train_features, train_labels, val_features, val_labels, epochs=50, lr=0.01):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(meta_model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in tqdm(range(epochs), desc="Training"):
        meta_model.train()
        optimizer.zero_grad()

        # Forward pass
        text_prob, image_prob = train_features[:, 0], train_features[:, 1]
        outputs = meta_model(text_prob, image_prob)

        # Compute loss
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        # Check weight updates
        print(f"Epoch {epoch + 1}: w_text = {meta_model.w_text.item():.4f}, w_image = {meta_model.w_image.item():.4f}")

        # Validation
        meta_model.eval()
        with torch.no_grad():
            val_text_prob, val_image_prob = val_features[:, 0], val_features[:, 1]
            val_outputs = meta_model(val_text_prob, val_image_prob)
            val_loss = criterion(val_outputs, val_labels)

        # Compute accuracy
        train_accuracy = compute_accuracy(meta_model, train_features, train_labels)
        val_accuracy = compute_accuracy(meta_model, val_features, val_labels)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save best weights
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_weights = {
                'w_text': meta_model.w_text.item(),
                'w_image': meta_model.w_image.item(),
            }

    return best_weights

# K-Fold Cross-Validation
def kfold_training(csv_path, k=5, epochs=50, lr=0.01):
    features, labels = load_data(csv_path)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_best_weights = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f"Fold {fold + 1}/{k}")

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]

        # Initialize meta-model for each fold
        meta_model = LearnableMajorityVoting()

        # Train the model and store the best weights
        best_weights = train_meta_model(meta_model, train_features, train_labels, val_features, val_labels, epochs, lr)
        all_best_weights.append(best_weights)

        # Validation report
        val_accuracy = compute_accuracy(meta_model, val_features, val_labels)
        print(f"Validation Accuracy for Fold {fold + 1}: {val_accuracy:.4f}")
        preds = (meta_model(val_features[:, 0], val_features[:, 1]) > 0.5).int()
        print(classification_report(val_labels.cpu(), preds.cpu(), target_names=['Happy', 'Depressed']))

    return all_best_weights

# Train-Test Split and Save Best Weights
def train_test_split_and_save(csv_path, test_size=0.2, epochs=50, lr=0.01, save_path="best_weights.pth"):
    features, labels = load_data(csv_path)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_size, random_state=42
    )

    # Initialize meta-model
    meta_model = LearnableMajorityVoting()

    # Train the model
    best_weights = train_meta_model(meta_model, train_features, train_labels, test_features, test_labels, epochs, lr)

    # Test report
    test_accuracy = compute_accuracy(meta_model, test_features, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    preds = (meta_model(test_features[:, 0], test_features[:, 1]) > 0.5).int()
    print(classification_report(test_labels.cpu(), preds.cpu(), target_names=['Happy', 'Depressed']))

    # Save best weights to a file
    torch.save(best_weights, save_path)
    print(f"Best weights saved to {save_path}")

# Example Usage
csv_path = "/home/borhan/Desktop/multimodal_depression_detection/data/texts/multimodal/probs/text_image_prob_merged.csv"
kfold_best_weights = kfold_training(csv_path, k=100, epochs=200, lr=0.1)
print("K-Fold Best Weights:", kfold_best_weights)

train_test_split_and_save(csv_path, test_size=0.2, epochs=200, lr=0.1, save_path="/home/borhan/Desktop/multimodal_depression_detection/checkpoints/multimodal/best_weights_multimodal.pth")
