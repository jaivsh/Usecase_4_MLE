"""
Inference module for sentiment analysis predictions
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
import pickle
import os
from nltk.tokenize import word_tokenize
from .trainer import TextCNN, CNNConfig

@dataclass
class InferenceConfig:
    """Configuration for model inference"""
    model_path: str = "models/text_cnn_sentiment_model.pth"
    vocab_path: str = "models/vocab.pkl"
    max_len: int = 128
    batch_size: int = 64
    device: str = "auto"  # "auto", "cpu", or "cuda"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

class SentimentPredictor:
    """Handles sentiment prediction using trained CNN model"""
    
    def __init__(self, inference_config: InferenceConfig, model_config: CNNConfig):
        self.inference_config = inference_config
        self.model_config = model_config
        self.device = torch.device(inference_config.device)
        self.model = None
        self.vocab = None
        self.label_map = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
        
        print(f"Inference device: {self.device}")
    
    def load_model_and_vocab(self):
        """Load trained model and vocabulary"""
        # Load vocabulary
        if not os.path.exists(self.inference_config.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {self.inference_config.vocab_path}")
        
        with open(self.inference_config.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        print(f"Loaded vocabulary with {len(self.vocab)} words")
        
        # Initialize model architecture
        self.model = TextCNN(
            vocab_size=len(self.vocab),
            embedding_dim=self.model_config.embedding_dim,
            n_filters=self.model_config.n_filters,
            filter_sizes=self.model_config.filter_sizes,
            output_dim=self.model_config.n_classes,
            dropout_rate=self.model_config.dropout_rate,
            pad_idx=self.vocab['<pad>']
        )
        
        # Load trained weights
        if not os.path.exists(self.inference_config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.inference_config.model_path}")
        
        self.model.load_state_dict(torch.load(self.inference_config.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully from {self.inference_config.model_path}")
    
    def preprocess_text(self, text: str) -> List[int]:
        """Preprocess text for inference"""
        if not isinstance(text, str):
            text = str(text)
        
        tokens = word_tokenize(text.lower())
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        # Pad or truncate to max_len
        if len(token_ids) < self.inference_config.max_len:
            token_ids += [self.vocab['<pad>']] * (self.inference_config.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.inference_config.max_len]
        
        return token_ids
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """Predict sentiment for a single text"""
        if self.model is None or self.vocab is None:
            self.load_model_and_vocab()
        
        # Preprocess text
        token_ids = self.preprocess_text(text)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get probabilities for all classes
        prob_dict = {
            self.label_map[i]: probabilities[0][i].item() 
            for i in range(len(self.label_map))
        }
        
        return {
            'text': text,
            'predicted_sentiment': self.label_map[predicted_class],
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float, Dict]]]:
        """Predict sentiment for a batch of texts"""
        if self.model is None or self.vocab is None:
            self.load_model_and_vocab()
        
        results = []
        
        # Process texts in batches
        for i in range(0, len(texts), self.inference_config.batch_size):
            batch_texts = texts[i:i + self.inference_config.batch_size]
            batch_token_ids = [self.preprocess_text(text) for text in batch_texts]
            input_tensor = torch.tensor(batch_token_ids, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy()
                confidences = probabilities.max(dim=1)[0].cpu().numpy()
            
            # Process batch results
            for j, text in enumerate(batch_texts):
                predicted_class = predicted_classes[j]
                confidence = confidences[j]
                
                prob_dict = {
                    self.label_map[k]: probabilities[j][k].item() 
                    for k in range(len(self.label_map))
                }
                
                results.append({
                    'text': text,
                    'predicted_sentiment': self.label_map[predicted_class],
                    'confidence': confidence,
                    'probabilities': prob_dict
                })
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'Cleaned_Comment_Text') -> pd.DataFrame:
        """Predict sentiment for texts in a DataFrame"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].astype(str).tolist()
        predictions = self.predict_batch(texts)
        
        # Add predictions to DataFrame
        df_result = df.copy()
        df_result['Predicted_Sentiment'] = [pred['predicted_sentiment'] for pred in predictions]
        df_result['Prediction_Confidence'] = [pred['confidence'] for pred in predictions]
        
        # Add probability columns
        for sentiment in self.label_map.values():
            df_result[f'Prob_{sentiment}'] = [pred['probabilities'][sentiment] for pred in predictions]
        
        return df_result
    
    def evaluate_predictions(self, df: pd.DataFrame, true_label_column: str = 'Simulated_Sentiment_Label') -> Dict:
        """Evaluate predictions against true labels"""
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        
        if 'Predicted_Sentiment' not in df.columns:
            df = self.predict_dataframe(df)
        
        if true_label_column not in df.columns:
            raise ValueError(f"True label column '{true_label_column}' not found in DataFrame")
        
        true_labels = df[true_label_column].tolist()
        predicted_labels = df['Predicted_Sentiment'].tolist()
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=list(self.label_map.values()))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(true_labels, predicted_labels, output_dict=True),
            'confusion_matrix': cm,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }
    
    def save_predictions(self, df: pd.DataFrame, output_path: str):
        """Save predictions to CSV file"""
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

def load_pretrained_predictor(model_path: str = "models/text_cnn_sentiment_model.pth",
                            vocab_path: str = "models/vocab.pkl") -> SentimentPredictor:
    """Convenience function to load a pretrained predictor"""
    inference_config = InferenceConfig(model_path=model_path, vocab_path=vocab_path)
    
    # Load model config (you might want to save this during training)
    model_config = CNNConfig()  # Using default config
    
    predictor = SentimentPredictor(inference_config, model_config)
    predictor.load_model_and_vocab()
    
    return predictor