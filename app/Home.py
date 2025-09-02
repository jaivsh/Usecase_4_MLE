# pylint: disable=invalid-name
"""
Streamlit app for Social Media Sentiment Analysis
"""
import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure page
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_status():
    """Check if the API backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

def call_api_predict(text: str) -> Optional[dict]:
    """Call API for single text prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def call_api_upload_csv(file_content: bytes, text_column: str) -> Optional[dict]:
    """Call API for CSV file upload"""
    try:
        files = {"file": ("data.csv", file_content, "text/csv")}
        data = {"text_column": text_column}
        response = requests.post(
            f"{API_BASE_URL}/upload_csv",
            files=files,
            data=data
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload Error: {str(e)}")
        return None

def call_api_upload_json(file_content: bytes, text_field: str) -> Optional[dict]:
    """Call API for JSON file upload"""
    try:
        files = {"file": ("data.json", file_content, "application/json")}
        data = {"text_field": text_field}
        response = requests.post(
            f"{API_BASE_URL}/upload_json",
            files=files,
            data=data
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload Error: {str(e)}")
        return None

def create_sentiment_chart(predictions_df: pd.DataFrame):
    """Create sentiment distribution chart"""
    if 'Predicted_Sentiment' in predictions_df.columns:
        sentiment_counts = predictions_df['Predicted_Sentiment'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c', 
                'Neutral': '#f39c12'
            }
        )
        return fig
    return None

def create_confidence_chart(predictions_df: pd.DataFrame):
    """Create confidence score distribution"""
    if 'Prediction_Confidence' in predictions_df.columns:
        fig = px.histogram(
            predictions_df,
            x='Prediction_Confidence',
            title="Prediction Confidence Distribution",
            nbins=20
        )
        return fig
    return None

def main():
    # Header
    st.title("📊 Social Media Sentiment Analysis")
    st.markdown("---")
    
    # Check API status
    if not check_api_status():
        st.error("🔴 API Backend is not running! Please start the FastAPI server.")
        st.code("python app/backend.py")
        return
    
    st.success("🟢 API Backend is running")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Single Text Prediction", "File Upload & Analysis", "Dataset Explorer"]
    )
    
    if app_mode == "Single Text Prediction":
        st.header("🔮 Single Text Prediction")
        
        # Text input
        user_text = st.text_area(
            "Enter text for sentiment analysis:",
            placeholder="Type your text here... (e.g., 'I love this product!')"
        )
        
        # Model selection (as per requirements)
        model_choice = st.selectbox(
            "Select Model:",
            ["CNN Model", "ALL"]  # You can expand this when you have multiple models
        )
        
        if st.button("Analyze Sentiment"):
            if user_text.strip():
                with st.spinner("Analyzing sentiment..."):
                    result = call_api_predict(user_text)
                    
                if result:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Sentiment", result['predicted_sentiment'])
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    
                    with col3:
                        sentiment_color = {
                            'Positive': 'green',
                            'Negative': 'red',
                            'Neutral': 'orange'
                        }
                        st.markdown(
                            f"<h3 style='color: {sentiment_color.get(result['predicted_sentiment'], 'gray')}'>"
                            f"Result: {result['predicted_sentiment']}</h3>",
                            unsafe_allow_html=True
                        )
                    
                    # Probability breakdown
                    st.subheader("Probability Breakdown")
                    prob_df = pd.DataFrame(
                        list(result['probabilities'].items()),
                        columns=['Sentiment', 'Probability']
                    )
                    
                    fig = px.bar(
                        prob_df,
                        x='Sentiment',
                        y='Probability',
                        color='Sentiment',
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Negative': '#e74c3c',
                            'Neutral': '#f39c12'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze.")
    
    elif app_mode == "File Upload & Analysis":
        st.header("📁 File Upload & Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'json'],
            help="Upload CSV or JSON file containing text data"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Column/field selection
            if file_type == 'csv':
                # Preview CSV to select column
                df_preview = pd.read_csv(uploaded_file)
                st.subheader("File Preview")
                st.dataframe(df_preview.head())
                
                text_column = st.selectbox(
                    "Select text column:",
                    df_preview.columns.tolist()
                )
                uploaded_file.seek(0)  # Reset file pointer
                
            else:  # JSON
                data_preview = json.load(uploaded_file)
                if isinstance(data_preview, list) and len(data_preview) > 0:
                    available_fields = list(data_preview[0].keys())
                    text_field = st.selectbox(
                        "Select text field:",
                        available_fields
                    )
                uploaded_file.seek(0)  # Reset file pointer
            
            # Model selection
            model_choice = st.selectbox(
                "Select Model:",
                ["CNN Model", "ALL"],
                key="file_model"
            )
            
            # Process file
            if st.button("Process File"):
                with st.spinner("Processing file and making predictions..."):
                    file_content = uploaded_file.read()
                    
                    if file_type == 'csv':
                        result = call_api_upload_csv(file_content, text_column)
                    else:
                        result = call_api_upload_json(file_content, text_field)
                    
                    if result:
                        st.success(f"✅ {result['message']}")
                        
                        # Convert predictions to DataFrame
                        predictions_df = pd.DataFrame(result['predictions'])
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(predictions_df)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_chart = create_sentiment_chart(predictions_df)
                            if sentiment_chart:
                                st.plotly_chart(sentiment_chart, use_container_width=True)
                        
                        with col2:
                            confidence_chart = create_confidence_chart(predictions_df)
                            if confidence_chart:
                                st.plotly_chart(confidence_chart, use_container_width=True)
                        
                        # Download results
                        csv_data = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_data,
                            file_name="sentiment_predictions.csv",
                            mime="text/csv"
                        )
    
    elif app_mode == "Dataset Explorer":
        st.header("🔍 Dataset Explorer")
        
        # Load processed dataset for exploration
        try:
            df = pd.read_csv("data/processed/processed_comments_with_sentiment.csv")
            
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Comments", len(df))
            with col2:
                st.metric("Unique Posts", df['Post_ID'].nunique())
            with col3:
                st.metric("Platforms", df['Platform'].nunique())
            with col4:
                st.metric("Avg Comment Length", f"{df['Cleaned_Comment_Text'].str.len().mean():.1f}")
            
            # Dataset sample
            st.subheader("Data Sample")
            display_columns = ['Comment_Text', 'Cleaned_Comment_Text', 'Simulated_Sentiment_Label', 'Platform']
            available_columns = [col for col in display_columns if col in df.columns]
            st.dataframe(df[available_columns].head(10))
            
            # Visualizations
            st.subheader("Data Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_dist = df['Simulated_Sentiment_Label'].value_counts()
                fig_sent = px.pie(
                    values=sentiment_dist.values,
                    names=sentiment_dist.index,
                    title="Overall Sentiment Distribution"
                )
                st.plotly_chart(fig_sent, use_container_width=True)
            
            with col2:
                # Platform distribution
                platform_dist = df['Platform'].value_counts()
                fig_platform = px.bar(
                    x=platform_dist.index,
                    y=platform_dist.values,
                    title="Comments by Platform"
                )
                st.plotly_chart(fig_platform, use_container_width=True)
            
            # Sentiment by platform
            st.subheader("Sentiment Analysis by Platform")
            sentiment_platform = df.groupby(['Platform', 'Simulated_Sentiment_Label']).size().reset_index(name='Count')
            fig_sent_platform = px.bar(
                sentiment_platform,
                x='Platform',
                y='Count',
                color='Simulated_Sentiment_Label',
                title="Sentiment Distribution Across Platforms"
            )
            st.plotly_chart(fig_sent_platform, use_container_width=True)
            
        except FileNotFoundError:
            st.error("Processed dataset not found. Please run data processing first.")
            st.code("python src/main.py --mode process")

if __name__ == "__main__":
    main()