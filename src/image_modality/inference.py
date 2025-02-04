import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.image.test_image_module import transform, classify_image, load_model
import torchvision


class ImageInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        self.transform = transform

    def inference_image(self, image_path: str):
        predicted_label, confidence, prediction = classify_image(image_path, self.model, self.transform)
        return predicted_label, confidence, prediction
    
    
if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "../../checkpoints/image_modality/best_image_model.pth")
    image_modality_checkpoint = os.path.abspath(config_path)
    image_inference = ImageInference(image_modality_checkpoint)
    image_path = os.path.join(script_dir, "../../data/images/happy/h_7.jpg")
    image_path = os.path.abspath(image_path)
    # predicted_label, confidence, prediction = image_inference.inference_image(image_path)
    predicted_label, confidence, prediction = image_inference.inference_image('/home/borhan/Desktop/multimodal_depression_detection/data/test/image/h_18.jpg')
    print(predicted_label, confidence, prediction)