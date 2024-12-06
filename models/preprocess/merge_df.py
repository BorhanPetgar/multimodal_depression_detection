import pandas as pd


def merge(csv1, csv2, multimodal_pair):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df3 = pd.read_csv(multimodal_pair)
    label = df3['label']
    merged_df = pd.concat([df1, df2, label], axis=1)

    merged_df.columns = ['text_prob', 'image_prob', 'label']

    merged_df.to_csv('/home/borhan/Desktop/multimodal_depression_detection/data/texts/multimodal/probs/text_image_prob_merged.csv', index=False)
    
if __name__ == '__main__':
    text_csv = '/home/borhan/Desktop/multimodal_depression_detection/data/texts/multimodal/probs/dep_prob_list.csv'
    image_csv = '/home/borhan/Desktop/multimodal_depression_detection/data/texts/multimodal/probs/image_prob_list.csv'
    multimodal_pair = '/home/borhan/Desktop/multimodal_depression_detection/data/texts/balanced_data3.csv'
    merge(text_csv, image_csv, multimodal_pair)