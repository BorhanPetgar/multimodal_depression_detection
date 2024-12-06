import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.image_modality.inference import ImageInference
import pandas as pd

if __name__ == '__main__':
    csv_path = '/home/borhan/Desktop/multimodal_depression_detection/data/texts/balanced_data3.csv'
    happy_dir = '/home/borhan/Desktop/multimodal_depression_detection/data/images/happy'
    sad_dir = '/home/borhan/Desktop/multimodal_depression_detection/data/images/sad'
    
    prob_list = []
    
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "../../checkpoints/image_modality/best_image_model.pth")
    image_modality_checkpoint = os.path.abspath(config_path)
    
    image_inference = ImageInference(image_modality_checkpoint)
    
    # image_path = os.path.join(script_dir, "../../data/images/happy/h_7.jpg")
    # image_path = os.path.abspath(image_path)
    # predicted_label, confidence, prediction = image_inference.inference_image(image_path)
    # print(predicted_label, confidence, prediction)
    
    df = pd.read_csv(csv_path)
    image_series = df['image']
    

    for image_path in image_series:
        if image_path.startswith('h'):
            image_abs_path = os.path.join(happy_dir, image_path)
            print(image_abs_path)
        elif image_path.startswith('s'):
            image_abs_path = os.path.join(sad_dir, image_path)

        predicted_label, confidence, prediction = image_inference.inference_image(image_abs_path)

        prob_list.append(prediction.cpu().numpy())
        
    
    column_of_probs = pd.DataFrame({'prob': prob_list})
    column_of_probs.to_csv(path_or_buf='/home/borhan/Desktop/multimodal_depression_detection/data/texts/multimodal/probs/image_prob_list.csv', index=False, columns=['prob'])

    