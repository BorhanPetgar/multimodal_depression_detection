"""
This script demonstrates how to use the trained image classification model to classify a single image.
It also provides an example of how to use the model to classify multiple images and calculate the accuracy.
It gets the images from the data/images/happy and data/images/sad directories.
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt


# Define the transform to match the training preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(model_path='/home/borhan/Desktop/multimodal_depression_detection/checkpoints/image_modality/best_image_model.pth'):
    """
    Load the pre-trained model.
    """
    model = models.resnet18(pretrained=True)

    # Modify the final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # model = models.resnet18(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 1),
    #     nn.Sigmoid()
    # )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()
    return model

# Load the trained model
model = load_model()

def classify_image(image_path, model, transform, show_image=False):
    """
    Classifies a single image as 'happy' or 'sad'.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move tensor to the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Convert prediction to binary label
    predicted_label = 'sad' if prediction.item() > 0.5 else 'happy'
    
    # Display the image and prediction
    if show_image:
        plt.imshow(image)
        plt.title(f'Prediction: {predicted_label} Confidence: {prediction.item():.2f}')
        plt.axis('off')
        plt.show()
    
    return predicted_label

# Example usage
import os
happy_path = '/home/borhan/Desktop/multimodal_depression_detection/data/images/happy'
sad_path = '/home/borhan/Desktop/multimodal_depression_detection/data/images/sad'


happy_count = 0
sad_count = 0

# image_root, _, image_paths = os.walk(happy_path)
for image_path in os.listdir(sad_path):
    print(100 *'*')
    # print(image_path)
    image_path = os.path.join(sad_path, image_path)  # Replace with your image path
    print(image_path)
    predicted_label = classify_image(image_path, model, transform)
    print(f'The image is classified as: {predicted_label}')
    if predicted_label == 'sad':
        sad_count += 1
    else:
        happy_count += 1
        
        
print(f'Happy count: {happy_count}')
print(f'Sad count: {sad_count}')
print(f'Total count: {happy_count + sad_count}')
print(f'acc: {sad_count / (happy_count + sad_count) * 100}%')
