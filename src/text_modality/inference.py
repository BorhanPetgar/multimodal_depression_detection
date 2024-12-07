import pandas as pd
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
        dep_prob_list = run_inference(text, self.model_path, self.glove_path)
        return dep_prob_list


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "../../checkpoints/text_modality/best_text_model_v3.pth")
    text_modality_checkpoint = os.path.abspath(config_path)
    
    glove_path = os.path.join(script_dir, "../../checkpoints/glove.6B/glove.6B.50d.txt")
    glove_path = os.path.abspath(glove_path)
    
    text_inference = TextInference(text_modality_checkpoint, glove_path)
    # text = '/home/borhan/Desktop/multimodal_depression_detection/models/text/my_examples.csv'
    # text = '/home/borhan/Desktop/multimodal_depression_detection/data/texts/multimodal/text.csv'
    text = '/home/borhan/Desktop/multimodal_depression_detection/data/test/text/text_only.csv'
    dep_prob_list = text_inference.inference_text(text)
    print(20 * '*')
    
    list_of_probs = []
    for tensor in dep_prob_list:
        for item in tensor:
            list_of_probs.append(item.cpu().numpy())
    
    
    column_of_probs = pd.DataFrame({'prob': list_of_probs})
    column_of_probs.to_csv(path_or_buf='/home/borhan/Desktop/multimodal_depression_detection/data/test/text/test_text_prob_list.csv', index=False, columns=['prob'])