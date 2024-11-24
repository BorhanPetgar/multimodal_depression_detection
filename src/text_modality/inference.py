from models.image.test_image_module import transform, classify_image, load_model
import torchvision
import os


class ImageInference:
    def __init__(self, model_path: str, transform: torchvision.transforms.Compose):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        self.transform = transform

    def inference_image(self, image_path: str):
        predicted_label = classify_image(image_path, self.model, self.transform)
        return predicted_label
    
if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "../../checkpoints/image_modality/best_image_model.yaml")
    config_path = os.path.abspath(config_path)
    image_modality_checkpoint = ''