import pandas as pd
import numpy as np

# Load the CSV data into a DataFrame


df = pd.read_csv('/home/borhan/Desktop/multimodal_depression_detection/data/texts/multimodal/probs/text_image_prob_merged.csv')

# Function to calculate accuracy based on given weight for text_prob
def calculate_accuracy(w1):
    w2 = 1 - w1
    num_rows = len(df)
    combined_prob = w1 * df['text_prob'] + w2 * df['image_prob']
    predictions = (combined_prob >= 0.5).astype(int)
    summation = (predictions == df['label']).sum()
    # accuracy2 = (predictions == df['label']).mean()
    acc = summation / num_rows
    return acc

# Hyperparameter tuning over a range of weights from 0 to 1
weights = np.linspace(0, 1, 10001)
accuracies = [calculate_accuracy(w) for w in weights]

# Find the best weight and accuracy
best_weight_index = np.argmax(accuracies)
best_weight = weights[best_weight_index]
best_accuracy = accuracies[best_weight_index]

# Display results
print(f"Best weight for text_prob: {best_weight:.3f}")
print(f"Best weight for image_prob: {1 - best_weight:.3f}")
print(f"Best accuracy: {best_accuracy:.13f}")