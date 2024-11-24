import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from copy import deepcopy

# Load dataset
train_df = pd.read_csv('/home/fteam5/borhan/nlp/project/merged_shuffled.csv')

# Load GloVe embeddings
words = dict()
def add_to_dict(d, filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            try:
                d[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue

add_to_dict(words, '/home/fteam5/borhan/nlp/project/glove/glove.6B.50d.txt')

# Tokenizer and Lemmatizer
nltk.download('wordnet')
tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = nltk.WordNetLemmatizer()

def message_to_word_vectors(message, word_dict=words):
    processed_list_of_tokens = message_to_token_list(message)
    vectors = [word_dict[token] for token in processed_list_of_tokens if token in word_dict]
    return np.array(vectors, dtype=float)

def message_to_token_list(s, word_dict=words):
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in word_dict]
    return useful_tokens



def pad_sequence(sequence, desired_length, vector_size=50):
    sequence_length = sequence.shape[0] if sequence.size else 0
    if sequence_length < desired_length:
        pad = np.zeros((desired_length - sequence_length, vector_size))
        if sequence_length == 0:
            sequence = sequence.reshape(0, vector_size)
        sequence = np.concatenate([sequence, pad])
    return sequence

# def preprocess_sentence(sentence, word_dict, tokenizer, lemmatizer, max_sequence_length=70):

#     word_vectors = message_to_word_vectors(sentence, word_dict)
#     padded_sequence = pad_sequence(word_vectors, max_sequence_length)
#     return torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(0)

class SentimentLSTM(nn.Module):
    def __init__(self):
        super(SentimentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=3, batch_first=True, dropout=0.2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 70, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def load_model(model_path):
    model = SentimentLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# def predict_sentence_class(sentence, model, word_dict, tokenizer, lemmatizer):
#     preprocessed_sentence = preprocess_sentence(sentence, word_dict, tokenizer, lemmatizer)
#     with torch.no_grad():
#         output = model(preprocessed_sentence)
#         prediction = (output.squeeze() > 0.5).int().item()
#     return prediction

def df_to_X_y(dff):
    y = dff['label'].to_numpy().astype(int)
    all_word_vector_sequences = [message_to_word_vectors(message) for message in dff['text']]
    all_word_vector_sequences = [seq if seq.shape[0] != 0 else np.zeros((1, 50)) for seq in all_word_vector_sequences]
    return all_word_vector_sequences, y

def pad_X(X, desired_sequence_length=70):
    X_copy = deepcopy(X)
    for i, x in enumerate(X):
        x_seq_len = x.shape[0]
        sequence_length_difference = desired_sequence_length - x_seq_len
        pad = np.zeros((sequence_length_difference, 50))
        X_copy[i] = np.concatenate([x, pad])
    return np.array(X_copy).astype(float)

# Split dataset
train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
split_index_1 = int(len(train_df) * 0.7)
split_index_2 = int(len(train_df) * 0.85)
train_df, val_df, test_df = train_df[:split_index_1], train_df[split_index_1:split_index_2], train_df[split_index_2:]

X_train, y_train = df_to_X_y(train_df)
X_val, y_val = df_to_X_y(val_df)
X_test, y_test = df_to_X_y(test_df)

X_train = pad_X(X_train)
X_val = pad_X(X_val)
X_test = pad_X(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the model
model = SentimentLSTM()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop with evaluation and saving correct/incorrect samples
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    correct_samples = []
    incorrect_samples = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item() * X_batch.size(0)

            preds = (outputs.squeeze() > 0.5).int()
            for i in range(len(preds)):
                if preds[i] == y_batch[i]:
                    correct_samples.append((X_batch[i].cpu().numpy(), y_batch[i].item(), preds[i].item()))
                else:
                    incorrect_samples.append((X_batch[i].cpu().numpy(), y_batch[i].item(), preds[i].item()))

    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model.pth')

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save correct and incorrect samples for this epoch
    correct_df = pd.DataFrame(correct_samples, columns=['Input', 'True Label', 'Predicted Label'])
    incorrect_df = pd.DataFrame(incorrect_samples, columns=['Input', 'True Label', 'Predicted Label'])

    correct_df.to_csv(f'correct_samples_epoch_{epoch+1}.csv', index=False)
    incorrect_df.to_csv(f'incorrect_samples_epoch_{epoch+1}.csv', index=False)

# Load the best model
model.load_state_dict(torch.load('model.pth'))

# Evaluate on test set
model.eval()
test_predictions = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        preds = (outputs.squeeze() > 0.5).int()
        test_predictions.extend(preds.cpu().numpy())

print(classification_report(y_test, test_predictions))


import torch

def preprocess_sentence(sentence, word_dict, tokenizer, lemmatizer, max_sequence_length=70):
    """
    Preprocess a sentence: tokenize, lemmatize, convert to GloVe embeddings, and pad sequence.
    """
    tokens = tokenizer.tokenize(sentence)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    word_vectors = [word_dict[token] for token in lemmatized_tokens if token in word_dict]

    # Convert to numpy array
    word_vectors = np.array(word_vectors, dtype=float)

    # Pad the sequence if needed
    if word_vectors.shape[0] < max_sequence_length:
        pad = np.zeros((max_sequence_length - word_vectors.shape[0], 50))
        word_vectors = np.concatenate([word_vectors, pad])
    else:
        word_vectors = word_vectors[:max_sequence_length]

    # Convert to PyTorch tensor
    return torch.tensor(word_vectors, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


def predict_sentence_class(sentence, model, word_dict, tokenizer, lemmatizer):
    """
    Predict the sentiment of a given sentence using the trained LSTM model.
    """
    # Preprocess the input sentence
    preprocessed_sentence = preprocess_sentence(sentence, word_dict, tokenizer, lemmatizer)

    # Ensure model is in evaluation mode and perform inference
    model.eval()
    with torch.no_grad():
        output = model(preprocessed_sentence)
        # Convert model output to binary prediction
        prediction = (output.squeeze() > 0.5).int().item()
        

    # Map the output to labels
    sentiment = "happy" if prediction == 1 else "sad"
    return sentiment, output


# Example usage:

# Load the trained model
model = SentimentLSTM()
model.load_state_dict(torch.load('model.pth'))

# Test the inference function with an example sentence
sentence = "I am feeling great today!"
sentiment = predict_sentence_class(sentence, model, words, tokenizer, lemmatizer)
print(f"Sentiment: {sentiment}")

# Another example
sentence = "I am very disappointed and sad."
sentiment = predict_sentence_class(sentence, model, words, tokenizer, lemmatizer)
print(f"Sentiment: {sentiment}")
 
