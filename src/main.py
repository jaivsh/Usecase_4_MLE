"""
Main CLI script for social media sentiment analysis pipeline
"""
import argparse
import sys
import os
from pathlib import Path
import pandas as pd
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data.processor import SocialMediaDataProcessor, DataProcessingConfig
from models.trainer import SentimentModelTrainer, CNNConfig
from models.inference import SentimentPredictor, InferenceConfig, load_pretrained_predictor

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")

def process_data(args):
    """Process raw data pipeline"""
    print("Starting data processing...")
    
    try:
        # Initialize data processing config
        config = DataProcessingConfig(
            raw_data_path=args.data_path,
            processed_data_path=args.processed_data_path,
            test_size=args.test_size,
            random_state=args.random_state,
            remove_stopwords=args.remove_stopwords
        )
        
        # Process data
        processor = SocialMediaDataProcessor(config)
        df_processed = processor.process_data()
        
        print(f"Data processing completed!")
        print(f"   - Processed {len(df_processed)} comments")
        print(f"   - Saved to: {config.processed_data_path}")
        
        # Show sentiment distribution
        sentiment_dist = df_processed['Simulated_Sentiment_Label'].value_counts()
        print(f"   - Sentiment distribution: {sentiment_dist.to_dict()}")
        
        return df_processed
        
    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        sys.exit(1)

def train_model(args):
    """Train sentiment analysis model"""
    print("Starting model training...")
    
    try:
        # Load processed data
        if not os.path.exists(args.processed_data_path):
            print(f"Processed data not found at {args.processed_data_path}")
            print("Run data processing first: python main.py --mode process")
            sys.exit(1)
        
        df = pd.read_csv(args.processed_data_path)
        print(f"Loaded {len(df)} processed comments")
        
        # Initialize training config
        config = CNNConfig(
            embedding_dim=args.embedding_dim,
            n_filters=args.n_filters,
            filter_sizes=args.filter_sizes,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            patience=args.patience,
            max_len=args.max_len,
            model_save_path=args.model_path,
            random_state=args.random_state
        )
        
        # Initialize trainer
        trainer = SentimentModelTrainer(config)
        
        # Build vocabulary from training data
        processor = SocialMediaDataProcessor(DataProcessingConfig())
        vocab, _ = processor.build_vocab(df['Cleaned_Comment_Text'].tolist())
        print(f"Built vocabulary with {len(vocab)} words")
        
        # Train model
        results = trainer.train_model(df, vocab)
        
        print(f"Training completed!")
        print(f"   - Test F1-Score: {results['test_f1']:.4f}")
        print(f"   - Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"   - Model saved to: {config.model_save_path}")
        
        return results
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        sys.exit(1)

def run_inference(args):
    """Run inference on new data"""
    print("Starting inference...")
    
    try:
        # Check if model exists
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}")
            print("Train a model first: python main.py --mode train")
            sys.exit(1)
        
        # Load data for inference
        if args.inference_data_path:
            df = pd.read_csv(args.inference_data_path)
            print(f"Loaded {len(df)} samples for inference")
        else:
            # Use test split from processed data
            df = pd.read_csv(args.processed_data_path)
            print(f"Using processed data for inference: {len(df)} samples")
        
        # Initialize inference
        inference_config = InferenceConfig(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            batch_size=args.batch_size
        )
        
        model_config = CNNConfig()  # Default config
        predictor = SentimentPredictor(inference_config, model_config)
        
        # Run predictions
        df_with_predictions = predictor.predict_dataframe(df, args.text_column)
        
        # Save results
        output_path = args.output_path or "results/predictions.csv"
        predictor.save_predictions(df_with_predictions, output_path)
        
        # Show sample results
        print("Sample predictions:")
        sample_cols = ['Comment_Text', 'Predicted_Sentiment', 'Prediction_Confidence']
        available_cols = [col for col in sample_cols if col in df_with_predictions.columns]
        print(df_with_predictions[available_cols].head())
        
        # Evaluate if true labels available
        if 'Simulated_Sentiment_Label' in df.columns:
            print("\nEvaluation Results:")
            evaluation = predictor.evaluate_predictions(df_with_predictions)
        
        print(f"Inference completed! Results saved to: {output_path}")
        
        return df_with_predictions
        
    except Exception as e:
        print(f"Error in inference: {str(e)}")
        sys.exit(1)

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Social Media Sentiment Analysis ML Pipeline")
    
    # Main mode argument
    parser.add_argument("--mode", 
                       choices=['process', 'train', 'inference', 'full_pipeline'],
                       required=True,
                       help="Pipeline mode to run")
    
    # Data arguments
    parser.add_argument("--data_path", 
                       default="data/raw/social_media_data.csv",
                       help="Path to raw data file")
    parser.add_argument("--processed_data_path",
                       default="data/processed/processed_comments_with_sentiment.csv", 
                       help="Path to processed data file")
    parser.add_argument("--inference_data_path",
                       help="Path to data for inference (optional)")
    parser.add_argument("--text_column",
                       default="Cleaned_Comment_Text",
                       help="Column name containing text for prediction")
    
    # Model arguments
    parser.add_argument("--model_path",
                       default="models/text_cnn_sentiment_model.pth",
                       help="Path to save/load model")
    parser.add_argument("--vocab_path",
                       default="models/vocab.pkl",
                       help="Path to save/load vocabulary")
    
    # Training hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=200,
                       help="Embedding dimension")
    parser.add_argument("--n_filters", type=int, default=128,
                       help="Number of CNN filters")
    parser.add_argument("--filter_sizes", type=int, nargs='+', default=[2, 3, 4],
                       help="CNN filter sizes")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3,
                       help="Early stopping patience")
    parser.add_argument("--max_len", type=int, default=128,
                       help="Maximum sequence length")
    
    # Other arguments
    parser.add_argument("--test_size", type=float, default=0.3,
                       help="Test set size for data splitting")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random state for reproducibility")
    parser.add_argument("--remove_stopwords", action='store_true',
                       help="Remove stopwords during preprocessing")
    parser.add_argument("--output_path",
                       help="Output path for inference results")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    print("Social Media Sentiment Analysis Pipeline")
    print(f"   Mode: {args.mode}")
    print(f"   Data: {args.data_path}")
    print("-" * 50)
    
    # Run pipeline based on mode
    try:
        if args.mode == 'process':
            process_data(args)
            
        elif args.mode == 'train':
            train_model(args)
            
        elif args.mode == 'inference':
            run_inference(args)
            
        elif args.mode == 'full_pipeline':
            print("Running full pipeline...")
            process_data(args)
            train_model(args) 
            run_inference(args)
            print("Full pipeline completed!")
        
        print("\nPipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()