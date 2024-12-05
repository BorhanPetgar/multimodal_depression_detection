import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.text.text_module import run_inference

import nltk
nltk.download('omw-1.4')


class TextInference:
    
    def __init__(self, model_path: str, glove_path: str):
        self.model_path = model_path
        self.glove_path = glove_path
        
    def inference_text(self, text: str):
        run_inference(text, self.model_path, self.glove_path)



if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "../../checkpoints/text_modality/best_text_model_v3.pth")
    text_modality_checkpoint = os.path.abspath(config_path)
    
    glove_path = os.path.join(script_dir, "../../checkpoints/glove.6B/glove.6B.50d.txt")
    glove_path = os.path.abspath(glove_path)
    
    text_inference = TextInference(text_modality_checkpoint, glove_path)
    # text = '/home/borhan/Desktop/multimodal_depression_detection/models/text/my_examples.csv'
    text = '/home/borhan/Desktop/multimodal_depression_detection/data/texts/multimodal/text.csv'
    text_inference.inference_text(text)
