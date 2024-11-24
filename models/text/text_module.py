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
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_to_dict(d, filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            try:
                d[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue

def message_to_word_vectors(message, word_dict):
    processed_list_of_tokens = message_to_token_list(message, word_dict)
    vectors = [word_dict[token] for token in processed_list_of_tokens if token in word_dict]
    return np.array(vectors, dtype=float)

def message_to_token_list(s, word_dict):
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
    model = SentimentLSTM().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def df_to_X_y(dff, words):
    y = dff['label'].to_numpy().astype(int)
    all_word_vector_sequences = [message_to_word_vectors(message, words) for message in dff['text']]
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
    return torch.tensor(word_vectors, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

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

def split_dataset(df: pd.DataFrame, val_size=0.15, test_size=0.15):
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    train_size = 1 - val_size - test_size
    split_index_1 = int(len(df) * train_size)
    split_index_2 = int(len(df) * (train_size + val_size))
    train_df, val_df, test_df = df[:split_index_1], df[split_index_1:split_index_2], df[split_index_2:]
    return train_df, val_df, test_df

def process_csv(df: pd.DataFrame, word_dict: dict, batch_size=32, shuffle=False):
    X, y = df_to_X_y(df, word_dict)
    X = pad_X(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def init_nltk():
    nltk.download('wordnet')
    
    global tokenizer, lemmatizer
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    lemmatizer = nltk.WordNetLemmatizer()
    
def init_vocab(glove_path: str):
    global words
    words = dict()
    add_to_dict(words, glove_path)

def inference(csv_path: str, model_path: str):
    model = SentimentLSTM().to(device)
    df = pd.read_csv(csv_path)
    model.load_state_dict(torch.load(model_path))

    # Evaluate on test set
    model.eval()
    
    loader = process_csv(df, words)
    X, y = df_to_X_y(df, words)
    
    mine_predictions = []
    with torch.no_grad():
        for X_batch, _ in loader:
            outputs = model(X_batch)
            preds = (outputs.squeeze() > 0.5).int()
            print(preds)
            mine_predictions.extend(preds.cpu().numpy())
            
    print(classification_report(y, mine_predictions))
    
def train_loop(train_loader, val_loader, test_loader, y_test):
    model = SentimentLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = float('inf')
    num_epochs = 20

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        with tqdm(train_loader, desc='Training', leave=False) as t:
            for X_batch, y_batch in t:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)

                # Calculate training accuracy
                preds = (outputs.squeeze() > 0.5).int()
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)
                
                t.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        correct_samples = []
        incorrect_samples = []

        with torch.no_grad():
            with tqdm(val_loader, desc='Validation', leave=False) as v:
                for X_batch, y_batch in v:
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    val_loss += loss.item() * X_batch.size(0)

                    # Calculate validation accuracy
                    preds = (outputs.squeeze() > 0.5).int()
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)

                    # Collect correct and incorrect samples
                    for i in range(len(preds)):
                        if preds[i] == y_batch[i]:
                            correct_samples.append((X_batch[i].cpu().numpy(), y_batch[i].item(), preds[i].item()))
                        else:
                            incorrect_samples.append((X_batch[i].cpu().numpy(), y_batch[i].item(), preds[i].item()))
                    
                    v.set_postfix({'loss': loss.item()})
            
        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / val_total

        # Check for improvement and save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_text_model_v4.pth')

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{num_epochs}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
    # Testing Phase
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        with tqdm(test_loader, desc='Testing', leave=False) as test_bar:
            for X_batch, y_batch in test_bar:
                outputs = model(X_batch)
                preds = (outputs.squeeze() > 0.5).int()
                test_predictions.extend(preds.cpu().numpy())
                test_targets.extend(y_batch.cpu().numpy())

    # Print classification report
    print('\nTest Classification Report:')
    print(classification_report(test_targets, test_predictions, digits=4))

def create_train_data_loader(csv_path: str):
    df = pd.read_csv(csv_path)
    
    train_df, val_df, test_df = split_dataset(df)
    
    X_test, y_test = df_to_X_y(test_df, words)
    
    train_loader = process_csv(train_df, words, batch_size=32)
    val_loader = process_csv(val_df, words,batch_size=32)
    test_loader = process_csv(test_df, words,batch_size=32)
    
    return train_loader, val_loader, test_loader, X_test, y_test
    
def train_model(csv_path: str):

    train_loader, val_loader, test_loader, X_test, y_test = create_train_data_loader(csv_path)
    
    train_loop(train_loader, val_loader, test_loader, y_test)
    
def run_train(glove_path: str, csv_path: str):
    init_nltk()
    init_vocab(glove_path)
    
    print('Training model...')
    train_model(csv_path)

def run_inference(csv_path: str, model_path: str, glove_path):
    init_nltk()
    init_vocab(glove_path)
    print('Running inference...')
    inference(csv_path, model_path)


if __name__ == "__main__":
    
    csv_path = '/home/fteam5/borhan/nlp/project/merged_shuffled.csv'
    my_csv_path = '/home/fteam5/borhan/nlp/project/my_examples.csv'
    glove_path = '/home/fteam5/borhan/nlp/project/glove/glove.6B.50d.txt'
    
    init_nltk()
    init_vocab(glove_path)
    
    print('Training model...')
    train_model(csv_path)

    print('Running inference...')
    inference(my_csv_path, 'best_text_model_v2.pth')

