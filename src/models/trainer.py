"""
Model training module for sentiment analysis CNN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
import os
import pickle

@dataclass
class CNNConfig:
    """Configuration for CNN training"""
    embedding_dim: int = 200
    n_filters: int = 128
    filter_sizes: List[int] = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    n_epochs: int = 15
    patience: int = 3
    max_len: int = 128
    n_classes: int = 3
    random_state: int = 42
    model_save_path: str = "models/text_cnn_sentiment_model.pth"
    vocab_save_path: str = "models/vocab.pkl"
    
    def __post_init__(self):
        if self.filter_sizes is None:
            self.filter_sizes = [2, 3, 4]

class TextDataset(Dataset):
    """Dataset class for text classification"""
    
    def __init__(self, texts: pd.Series, labels: pd.Series, vocab: dict, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        
        tokens = word_tokenize(str(text))
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<pad>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextCNN(nn.Module):
    """CNN model for text classification"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, 
                 filter_sizes: List[int], output_dim: int, dropout_rate: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, sent len]
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

class SentimentModelTrainer:
    """Handles CNN model training for sentiment analysis"""
    
    def __init__(self, config: CNNConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set random seeds
        np.random.seed(config.random_state)
        torch.manual_seed(config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_state)
    
    def prepare_data(self, df: pd.DataFrame, vocab: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test data loaders"""
        X = df['Cleaned_Comment_Text']
        y = df['Sentiment_ID']
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.config.random_state, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.config.random_state, stratify=y_temp
        )
        
        print(f"Train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # Create datasets and data loaders
        train_dataset = TextDataset(X_train, y_train, vocab, self.config.max_len)
        val_dataset = TextDataset(X_val, y_val, vocab, self.config.max_len)
        test_dataset = TextDataset(X_test, y_test, vocab, self.config.max_len)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model: nn.Module, data_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        progress_bar = tqdm(data_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        return total_loss / len(data_loader)
    
    def evaluate_epoch(self, model: nn.Module, data_loader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float, float, List, List]:
        """Evaluate model on validation/test set"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1, all_preds, all_labels
    
    def plot_training_metrics(self, train_losses: List[float], val_losses: List[float], 
                            val_accuracies: List[float]):
        """Plot training metrics"""
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'o-', label='Training Loss')
        plt.plot(epochs, val_losses, 'o-', label='Validation Loss')
        plt.title('CNN Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy', color='orange')
        plt.title('CNN Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    def train_model(self, df: pd.DataFrame, vocab: dict) -> Dict:
        """Complete training pipeline"""
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(df, vocab)
        
        # Initialize model
        model = TextCNN(
            vocab_size=len(vocab),
            embedding_dim=self.config.embedding_dim,
            n_filters=self.config.n_filters,
            filter_sizes=self.config.filter_sizes,
            output_dim=self.config.n_classes,
            dropout_rate=self.config.dropout_rate,
            pad_idx=vocab['<pad>']
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_f1 = 0
        patience_counter = 0
        
        print(f"\nStarting training for {self.config.n_epochs} epochs...")
        
        for epoch in range(self.config.n_epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_accuracy, val_f1, _, _ = self.evaluate_epoch(model, val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{self.config.n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            print("-" * 50)
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.config.model_save_path)
                print(f"New best model saved with F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Plot training metrics
        self.plot_training_metrics(train_losses, val_losses, val_accuracies)
        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(self.config.model_save_path))
        test_loss, test_accuracy, test_f1, test_preds, test_labels = self.evaluate_epoch(
            model, test_loader, criterion
        )
        
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        # Print classification report
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=sentiment_labels))
        
        # Save vocabulary
        with open(self.config.vocab_save_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {self.config.vocab_save_path}")
        
        return {
            'model': model,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'test_predictions': test_preds,
            'test_labels': test_labels,
            'vocab': vocab
        }