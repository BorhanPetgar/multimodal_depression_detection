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


class SentimentAnalysisModel:
    def __init__(self, glove_path='glove.6B.50d.txt', batch_size=32, sequence_length=70, learning_rate=0.0001, num_epochs=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Load GloVe embeddings
        self.words = self._load_glove_embeddings(glove_path)
        
        # Initialize tokenizer and lemmatizer
        nltk.download('wordnet', quiet=True)
        self.tokenizer = nltk.RegexpTokenizer(r"\w+")
        self.lemmatizer = nltk.WordNetLemmatizer()

        # Initialize model, criterion, and optimizer
        self.model = self._initialize_model()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _load_glove_embeddings(self, glove_path):
        words = {}
        with open(glove_path, 'r') as f:
            for line in f.readlines():
                line = line.split(' ')
                try:
                    words[line[0]] = np.array(line[1:], dtype=float)
                except:
                    continue
        return words

    def message_to_token_list(self, s):
        tokens = self.tokenizer.tokenize(s)
        lowercased_tokens = [t.lower() for t in tokens]
        lemmatized_tokens = [self.lemmatizer.lemmatize(t) for t in lowercased_tokens]
        useful_tokens = [t for t in lemmatized_tokens if t in self.words]
        return useful_tokens

    def message_to_word_vectors(self, message):
        processed_tokens = self.message_to_token_list(message)
        vectors = [self.words[token] for token in processed_tokens if token in self.words]
        return np.array(vectors, dtype=float)

    def pad_X(self, X):
        X_copy = deepcopy(X)
        for i, x in enumerate(X):
            x_seq_len = x.shape[0]
            if x_seq_len < self.sequence_length:
                pad = np.zeros((self.sequence_length - x_seq_len, 50))
                X_copy[i] = np.concatenate([x, pad])
            else:
                X_copy[i] = x[:self.sequence_length]
        return np.array(X_copy).astype(float)

    def df_to_X_y(self, df):
        y = df['label'].to_numpy().astype(int)
        all_word_vector_sequences = [self.message_to_word_vectors(message) for message in df['data']]
        all_word_vector_sequences = [seq if seq.shape[0] != 0 else np.zeros((1, 50)) for seq in all_word_vector_sequences]
        return all_word_vector_sequences, y

    def create_data_loader(self, df):
        X, y = self.df_to_X_y(df)
        X_padded = self.pad_X(X)
        X_tensor = torch.tensor(X_padded, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _initialize_model(self):
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
        
        model = SentimentLSTM().to(self.device)
        return model

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            val_loss = self.evaluate(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)
        return val_loss / len(data_loader.dataset)

    def test(self, test_loader):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        test_predictions = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                preds = (outputs.squeeze() > 0.5).int()
                test_predictions.extend(preds.cpu().numpy())
        return test_predictions

    def classification_report(self, y_true, y_pred):
        print("Classification Report:\n", classification_report(y_true, y_pred))

# Usage
if __name__ == "__main__":
    # Load dataset
    train_df = pd.read_csv('/content/merged_shuffled.csv')

    # Split dataset into train, validation, and test sets
    train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
    split_index_1 = int(len(train_df) * 0.7)
    split_index_2 = int(len(train_df) * 0.85)
    train_df, val_df, test_df = train_df[:split_index_1], train_df[split_index_1:split_index_2], train_df[split_index_2:]

    # Initialize and train the model
    model = SentimentAnalysisModel()
    train_loader = model.create_data_loader(train_df)
    val_loader = model.create_data_loader(val_df)
    test_loader = model.create_data_loader(test_df)

    model.train(train_loader, val_loader)

    # Evaluate on the test set
    y_test = test_df['label'].to_numpy().astype(int)
    test_predictions = model.test(test_loader)
    model.classification_report(y_test, test_predictions)
