import os
import pandas as pd
import random

# Step 1: Count the Number of Images in Each Folder
def count_images(image_root):
    happy_images = os.listdir(os.path.join(image_root, 'happy'))
    sad_images = os.listdir(os.path.join(image_root, 'sad'))
    return len(happy_images), len(sad_images), happy_images, sad_images

# Step 2: Randomly Sample Text Data from the CSV
def sample_text_data(csv_file, sample_size, label):
    df = pd.read_csv(csv_file)
    sampled_df = df.loc[df['label'] == label].sample(n=sample_size, random_state=42)
    return sampled_df

def assign_labels_and_shuffle(happy_images, sad_images, sampled_text_data) -> list:
    # Separate sampled texts by their labels
    happy_texts = sampled_text_data[sampled_text_data['label'] == 0]['text'].tolist()
    # happy_texts = sampled_text_data[sampled_text_data['label'] == 0]
    # print(happy_texts)
    # exit()
    sad_texts = sampled_text_data[sampled_text_data['label'] == 1]['text'].tolist()
    
    assert len(happy_images) == len(happy_texts), (
        f"Mismatch: {len(happy_images)} happy images but {len(happy_texts)} happy texts"
    )
    assert len(sad_images) == len(sad_texts), (
        f"Mismatch: {len(sad_images)} sad images but {len(sad_texts)} sad texts"
    )
    
    # Randomly shuffle the image and text data
    random.shuffle(happy_images)
    random.shuffle(sad_images)
    random.shuffle(happy_texts)
    random.shuffle(sad_texts)
    
    # Combine data into final list with assigned labels
    final_data = []
    
    # Assign texts to happy images with label 0
    for img in happy_images:
        text = happy_texts.pop(0) if happy_texts else ''
        final_data.append((img, text, 0))  # Label 0 for happy
    
    # Assign texts to sad images with label 1
    for img in sad_images:
        text = sad_texts.pop(0) if sad_texts else ''
        final_data.append((img, text, 1))  # Label 1 for sad
    
    # Shuffle final dataset to mix happy and sad samples
    random.shuffle(final_data)
    
    return final_data

# Step 4: Save to CSV
def save_to_csv(final_data, output_csv):
    image_paths = [image_path for image_path, _, _ in final_data]
    texts = [text for _, text, _ in final_data]
    labels = [label for _, _, label in final_data]
    
    df = pd.DataFrame({
        'image': image_paths,
        'text': texts,
        'label': labels
    })
    
    df.to_csv(output_csv, index=False)
    print(f"New CSV file created: {output_csv}")

def create_mixed_dataset(csv_file, image_root, output_csv):
    happy_count, sad_count, happy_images, sad_images = count_images(image_root)
    # print(happy_count, sad_count, happy_images[0])
    # exit()
    happy_sampled_text_data = sample_text_data(csv_file, happy_count, 0)
    sad_sampled_text_data = sample_text_data(csv_file, sad_count, 1)
    final_text_data = pd.concat([happy_sampled_text_data, sad_sampled_text_data])
    # print(final_text_data)
    # exit()
    # print(happy_sampled_text_data.shape, happy_count, sad_sampled_text_data.shape, sad_count)
    final_data = assign_labels_and_shuffle(happy_images, sad_images, final_text_data)
    print(final_data)
    save_to_csv(final_data, output_csv)

# Usage
create_mixed_dataset(
    csv_file='/home/borhan/Desktop/papers_to_read/project/2024-11-12/merged_shuffled.csv',
    image_root='/home/borhan/Desktop/papers_to_read/project/data/final',
    output_csv='/home/borhan/Desktop/papers_to_read/project/2024-11-12/balanced_data3.csv'
)
