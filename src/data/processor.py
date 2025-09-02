"""
Data processing module for social media sentiment analysis
"""
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

@dataclass
class DataProcessingConfig:
    """Configuration for data processing"""
    raw_data_path: str = "data/raw/social_media_data.csv"
    processed_data_path: str = "data/processed/processed_comments_with_sentiment.csv"
    test_size: float = 0.3
    validation_size: float = 0.5
    random_state: int = 42
    min_vocab_freq: int = 5
    max_sequence_length: int = 128
    remove_stopwords: bool = False

class SocialMediaDataProcessor:
    """Handles data preprocessing for social media analytics"""
    
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_label_map = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
        
    def load_data(self) -> pd.DataFrame:
        """Load raw social media data"""
        return pd.read_csv(self.config.raw_data_path)
    
    def extract_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract individual comments from comment columns"""
        comment_cols = [f'Comment {i}' for i in range(1, 11)]
        comment_data = []

        for index, row in df.iterrows():
            post_id = row['Post_ID']
            post_content = row['Post_Content']
            platform = row['Platform']
            
            engagement_ad_metrics = {
                col: row[col] for col in df.columns 
                if col not in ['Post_ID', 'Post_Content', 'Platform'] 
                and not col.startswith('Comment')
            }

            for i, col in enumerate(comment_cols):
                comment_text = row[col]
                if pd.notna(comment_text): 
                    comment_entry = {
                        'Post_ID': post_id,
                        'Comment_Index': i + 1,
                        'Comment_Text': comment_text,
                        'Post_Content': post_content, 
                        'Platform': platform,
                        **engagement_ad_metrics 
                    }
                    comment_data.append(comment_entry)

        return pd.DataFrame(comment_data)
    
    def clean_text(self, text: str) -> str:
        """Clean text data"""
        text = str(text).lower() 
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', lambda match: match.group(0)[1:], text) 
        text = text.translate(str.maketrans('', '', string.punctuation.replace('_', '')))  
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def handle_emojis(self, text: str) -> str:
        """Convert emojis to text"""
        return emoji.demojize(text, delimiters=(" ", " "))

    def preprocess_comment(self, text: str) -> str:
        """Preprocess individual comment"""
        text = self.handle_emojis(text)
        text = self.clean_text(text)
        
        if self.config.remove_stopwords:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
            text = ' '.join(tokens)
            
        return text
    
    def get_vader_sentiment(self, text: str) -> float:
        """Get VADER sentiment score"""
        return self.analyzer.polarity_scores(text)['compound']

    def assign_sentiment_label(self, compound_score: float) -> str:
        """Convert VADER score to sentiment label"""
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def process_data(self) -> pd.DataFrame:
        """Complete data processing pipeline"""
        # Load data
        df = self.load_data()
        print(f"Loaded data shape: {df.shape}")
        
        # Extract comments
        df_comments = self.extract_comments(df)
        print(f"Extracted comments shape: {df_comments.shape}")
        
        # Clean comments
        df_comments['Cleaned_Comment_Text'] = df_comments['Comment_Text'].apply(
            self.preprocess_comment
        )
        
        # Add sentiment analysis
        df_comments['VADER_Compound_Score'] = df_comments['Cleaned_Comment_Text'].apply(
            self.get_vader_sentiment
        )
        df_comments['Simulated_Sentiment_Label'] = df_comments['VADER_Compound_Score'].apply(
            self.assign_sentiment_label
        )
        df_comments['Sentiment_ID'] = df_comments['Simulated_Sentiment_Label'].map(
            self.sentiment_label_map
        )
        
        # Save processed data
        df_comments.to_csv(self.config.processed_data_path, index=False)
        print(f"Processed data saved to {self.config.processed_data_path}")
        
        return df_comments
    
    def build_vocab(self, texts: List[str]) -> tuple:
        """Build vocabulary from texts"""
        all_words = []
        for text in texts:
            all_words.extend(word_tokenize(text))
        
        word_counts = Counter(all_words)
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.config.min_vocab_freq]
        
        vocab = {'<pad>': 0, '<unk>': 1}
        for word in sorted(filtered_words): 
            if word not in vocab:
                vocab[word] = len(vocab)
        
        idx_to_word = {idx: word for word, idx in vocab.items()}
        return vocab, idx_to_word