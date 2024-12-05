import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.text_modality.inference import TextInference
from src.image_modality.inference import ImageInference

# Define the meta-model
class WeightedMajorityVoting(nn.Module):
    def __init__(self):
        super(WeightedMajorityVoting, self).__init__()
        self.weight_text = nn.Parameter(torch.tensor(0.5))  # Learnable weight for text model
        self.weight_image = nn.Parameter(torch.tensor(0.5))  # Learnable weight for image model

    def forward(self, prob_text, prob_image):
        weighted_sum = self.weight_text * prob_text + self.weight_image * prob_image
        return torch.sigmoid(weighted_sum)
    

# Prepare data for training the meta-model
def prepare_data(data: str = "../../data/texts/balanced_data3.csv"):
    script_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(script_dir, data))
    df = pd.read_csv(data_path)
    meta_data = []
    meta_labels = []

    for _, row in df.iterrows():
        # Text model prediction
        print(row['text'])
        exit()
        prob_text = classify_text(row['text'], text_model, text_transform, device)
        
        # Image model prediction
        image_path = f"data/images/{'happy' if row['image_path'].startswith('h') else 'sad'}/{row['image_path']}"
        prob_image, _, _ = classify_image(image_path, image_model, image_transform)
        
        meta_data.append((prob_text, prob_image))
        meta_labels.append(row['label'])
    
    return torch.tensor(meta_data, dtype=torch.float32), torch.tensor(meta_labels, dtype=torch.float32)

# Train the meta-model
def train_meta_model(meta_model, meta_data, meta_labels, epochs=10, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(meta_model.parameters(), lr=lr)

    for epoch in range(epochs):
        meta_model.train()
        optimizer.zero_grad()

        # Forward pass
        prob_text, prob_image = meta_data[:, 0], meta_data[:, 1]
        outputs = meta_model(prob_text, prob_image)
        
        # Compute loss
        loss = criterion(outputs, meta_labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

def load_models(
        text_model_checkpoint_path: str = "../../checkpoints/text_modality/best_text_model_v3.pth",
        image_model_checkpoint_path: str = "../../checkpoints/glove.6B/glove.6B.50d.txt",
        glove_file_path: str = "../../checkpoints/image_modality/best_image_model.pth",
        ):
    script_dir = os.path.dirname(__file__)
    text_modality_checkpoint = os.path.abspath(os.path.join(script_dir, text_model_checkpoint_path))
    glove_path = os.path.abspath(os.path.join(script_dir, glove_file_path))
    text_inference = TextInference(text_modality_checkpoint, glove_path)

    image_modality_checkpoint = os.path.abspath(os.path.join(script_dir, image_model_checkpoint_path))
    image_inference = ImageInference(image_modality_checkpoint)

    return text_inference, image_inference



if __name__ == "__main__":
    # text_inference, image_inference = load_models()
    prepare_data()
    exit()
    # Example Usage
    df = pd.read_csv("data.csv")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Placeholder models and transforms
    text_model = SentimentLSTM().to(device)  # Load your actual text model
    image_model = torch.load("image_model.pth").to(device)  # Load your actual image model
    text_transform = None  # Define preprocessing for text if needed
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    meta_model = WeightedMajorityVoting().to(device)

    meta_data, meta_labels = prepare_data(df, text_model, image_model, text_transform, image_transform, device)
    train_meta_model(meta_model, meta_data, meta_labels)
