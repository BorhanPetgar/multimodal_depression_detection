import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.image.test_image_module import transform, classify_image, load_model
import torchvision


class ImageInference:
    def __init__(self, model_path: str, transform: torchvision.transforms.Compose):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        self.transform = transform

    def inference_image(self, image_path: str):
        predicted_label, confidence = classify_image(image_path, self.model, self.transform)
        return predicted_label, confidence
    
    
if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "../../checkpoints/image_modality/best_image_model.pth")
    image_modality_checkpoint = os.path.abspath(config_path)
    image_inference = ImageInference(image_modality_checkpoint, transform)
    image_path = os.path.join(script_dir, "../../data/images/happy/h_7.jpg")
    image_path = os.path.abspath(image_path)
    predicted_label, confidence = image_inference.inference_image(image_path)
    print(predicted_label, confidence)