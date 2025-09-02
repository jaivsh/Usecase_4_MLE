"""
FastAPI backend for sentiment analysis predictions
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import pandas as pd
import json
import io
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.inference import load_pretrained_predictor, SentimentPredictor, InferenceConfig
from models.trainer import CNNConfig

app = FastAPI(title="Social Media Sentiment Analysis API", version="1.0.0")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[SentimentPredictor] = None

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    text: str
    predicted_sentiment: str
    confidence: float
    probabilities: Dict[str, float]

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global predictor
    try:
        predictor = load_pretrained_predictor(
            model_path="models/text_cnn_sentiment_model.pth",
            vocab_path="models/vocab.pkl"
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        predictor = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Social Media Sentiment Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if predictor is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    """Predict sentiment for a single text"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        result = predictor.predict_single(input_data.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(input_data: BatchTextInput):
    """Predict sentiment for multiple texts"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = predictor.predict_batch(input_data.texts)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), text_column: str = "text"):
    """Upload CSV file and get predictions"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
            )
        
        # Get predictions
        df_with_predictions = predictor.predict_dataframe(df, text_column)
        
        # Convert to JSON format
        result = df_with_predictions.to_dict(orient='records')
        
        return {
            "message": f"Processed {len(df)} rows",
            "columns": list(df_with_predictions.columns),
            "predictions": result[:100]  # Limit to first 100 for response size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

@app.post("/upload_json") 
async def upload_json(file: UploadFile = File(...), text_field: str = "text"):
    """Upload JSON file and get predictions"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="File must be a JSON")
    
    try:
        # Read JSON file
        contents = await file.read()
        data = json.loads(contents.decode('utf-8'))
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if text_field not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Field '{text_field}' not found. Available fields: {list(df.columns)}"
            )
        
        # Get predictions
        df_with_predictions = predictor.predict_dataframe(df, text_field)
        result = df_with_predictions.to_dict(orient='records')
        
        return {
            "message": f"Processed {len(df)} records", 
            "predictions": result[:100]  # Limit to first 100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON processing error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "CNN",
        "vocabulary_size": len(predictor.vocab) if predictor.vocab else 0,
        "classes": list(predictor.label_map.values()),
        "max_sequence_length": predictor.inference_config.max_len
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)