#!/usr/bin/env python3
"""
Market AI Web Interface
Interactive dashboard for financial market prediction models
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta
import logging

# Import our modules
try:
    from advanced_models import create_advanced_model, FinancialTransformer
    from data_loader_hf import load_news_hf
    from sentiment import analyze_news_sentiment
    from features_enhanced import add_technical_indicators
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Market AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stock_data():
    """Load stock price data"""
    stock_files = ['AAPL.csv', 'MSFT.csv', 'TSLA.csv', 'GOOGL.csv', 'AMZN.csv', 
                   'META.csv', 'NVDA.csv', 'AMD.csv', 'INTC.csv', 'QCOM.csv']
    
    all_data = []
    for file in stock_files:
        path = f'data/{file}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Ticker'] = file.replace('.csv', '')
            df['Date'] = pd.to_datetime(df['Date'])
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def load_news_data():
    """Load news data"""
    if os.path.exists('data/news.csv'):
        df = pd.read_csv('data/news.csv')
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        return df
    return pd.DataFrame()

@st.cache_data
def load_model_results():
    """Load model comparison results"""
    results_path = 'experiments/financial_models_comparison/model_comparison.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}

def create_price_chart(stock_data, selected_ticker):
    """Create interactive price chart"""
    if stock_data.empty:
        return go.Figure()
    
    ticker_data = stock_data[stock_data['Ticker'] == selected_ticker].sort_values('Date')
    
    if ticker_data.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f'{selected_ticker} Stock Price', 'Volume'],
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=ticker_data['Date'],
            open=ticker_data['Open'],
            high=ticker_data['High'],
            low=ticker_data['Low'],
            close=ticker_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=ticker_data['Date'],
            y=ticker_data['Volume'],
            name='Volume',
            marker_color='rgba(0,100,200,0.3)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{selected_ticker} Stock Analysis',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def create_sentiment_chart(news_data, selected_ticker):
    """Create sentiment analysis chart"""
    if news_data.empty:
        return go.Figure()
    
    if selected_ticker != 'ALL':
        news_data = news_data[news_data['ticker'] == selected_ticker]
    
    # Daily sentiment aggregation
    daily_sentiment = news_data.groupby(news_data['publishedAt'].dt.date).agg({
        'sentiment_score': ['mean', 'count']
    }).reset_index()
    
    daily_sentiment.columns = ['date', 'sentiment_mean', 'news_count']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=['Daily Sentiment Score', 'News Article Count']
    )
    
    # Sentiment line
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['sentiment_mean'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    
    # Add neutral line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # News count bars
    fig.add_trace(
        go.Bar(
            x=daily_sentiment['date'],
            y=daily_sentiment['news_count'],
            name='News Count',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'News Sentiment Analysis - {selected_ticker}',
        height=500
    )
    
    return fig

def create_model_comparison_chart(results):
    """Create model performance comparison chart"""
    if not results:
        return go.Figure()
    
    models = list(results.keys())
    val_losses = [results[model]['best_val_loss'] for model in models]
    val_accs = [results[model]['final_val_acc'] for model in models]
    param_counts = [results[model]['param_count'] for model in models]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Validation Loss by Model',
            'Validation Accuracy by Model', 
            'Model Parameters vs Performance',
            'Model Rankings'
        ]
    )
    
    # Validation loss
    fig.add_trace(
        go.Bar(x=models, y=val_losses, name='Val Loss', marker_color='red'),
        row=1, col=1
    )
    
    # Validation accuracy
    fig.add_trace(
        go.Bar(x=models, y=val_accs, name='Val Accuracy', marker_color='green'),
        row=1, col=2
    )
    
    # Parameters vs performance
    fig.add_trace(
        go.Scatter(
            x=param_counts, 
            y=val_losses,
            mode='markers+text',
            text=models,
            textposition="top center",
            name='Efficiency',
            marker=dict(size=10, color='blue')
        ),
        row=2, col=1
    )
    
    # Model rankings (score = accuracy / loss)
    scores = [acc / loss for acc, loss in zip(val_accs, val_losses)]
    sorted_data = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
    ranked_models, ranked_scores = zip(*sorted_data)
    
    fig.add_trace(
        go.Bar(
            x=list(ranked_scores),
            y=list(ranked_models),
            orientation='h',
            name='Ranking',
            marker_color='purple'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title="Model Performance Comparison")
    fig.update_xaxes(title_text="Models", row=1, col=1)
    fig.update_xaxes(title_text="Models", row=1, col=2)
    fig.update_xaxes(title_text="Parameters", row=2, col=1)
    fig.update_xaxes(title_text="Score", row=2, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Models", row=2, col=2)
    
    return fig

def main():
    """Main application"""
    st.title("📈 Market AI Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Market Data", "News Analysis", "Model Performance", "Live Prediction"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        stock_data = load_stock_data()
        news_data = load_news_data()
        model_results = load_model_results()
    
    if page == "Overview":
        st.header("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Stock Tickers", len(stock_data['Ticker'].unique()) if not stock_data.empty else 0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
            st.metric("News Articles", len(news_data) if not news_data.empty else 0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card warning-card">', unsafe_allow_html=True)
            st.metric("Models Tested", len(model_results) if model_results else 0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            best_acc = max([r['final_val_acc'] for r in model_results.values()]) if model_results else 0
            st.metric("Best Accuracy", f"{best_acc:.3f}" if best_acc else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System architecture
        st.subheader("🏗️ System Architecture")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Data Sources:**
            - 10 Major Tech Stocks (AAPL, MSFT, etc.)
            - 50K Financial News Articles (HuggingFace)
            - Real-time Technical Indicators
            - FinBERT Sentiment Analysis
            """)
        
        with col2:
            st.markdown("""
            **🤖 Model Features:**
            - 6-Layer Financial Transformer
            - Multi-task Learning (Returns + Volatility)
            - Advanced Technical Indicators (23+)
            - Attention-based News Integration
            """)
        
        # Recent performance
        if model_results:
            st.subheader("🏆 Top Performing Models")
            
            # Sort models by performance
            sorted_models = sorted(
                model_results.items(), 
                key=lambda x: x[1]['final_val_acc'] / x[1]['best_val_loss'], 
                reverse=True
            )
            
            for i, (model_name, results) in enumerate(sorted_models[:3]):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{i+1}. {model_name.replace('_', ' ').title()}**")
                with col2:
                    st.write(f"Accuracy: {results['final_val_acc']:.3f}")
                with col3:
                    st.write(f"Loss: {results['best_val_loss']:.4f}")
                with col4:
                    st.write(f"Params: {results['param_count']:,}")
    
    elif page == "Market Data":
        st.header("📈 Market Data Analysis")
        
        if stock_data.empty:
            st.error("No stock data available. Please check data files.")
            return
        
        # Ticker selection
        tickers = sorted(stock_data['Ticker'].unique())
        selected_ticker = st.selectbox("Select Stock Ticker", tickers)
        
        # Price chart
        st.subheader(f"{selected_ticker} Price Analysis")
        price_chart = create_price_chart(stock_data, selected_ticker)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Statistics
        ticker_data = stock_data[stock_data['Ticker'] == selected_ticker]
        if not ticker_data.empty:
            ticker_data = ticker_data.sort_values('Date')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${ticker_data['Close'].iloc[-1]:.2f}")
            with col2:
                daily_return = ticker_data['Close'].pct_change().iloc[-1]
                st.metric("Daily Return", f"{daily_return:.2%}")
            with col3:
                volatility = ticker_data['Close'].pct_change().std() * np.sqrt(252)
                st.metric("Annual Volatility", f"{volatility:.2%}")
            with col4:
                avg_volume = ticker_data['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    elif page == "News Analysis":
        st.header("📰 News Sentiment Analysis")
        
        if news_data.empty:
            st.error("No news data available. Please run fetch_news_once.py first.")
            return
        
        # Ticker selection for news
        news_tickers = ['ALL'] + sorted(news_data['ticker'].unique())
        selected_ticker = st.selectbox("Select Ticker for News", news_tickers)
        
        # Sentiment chart
        sentiment_chart = create_sentiment_chart(news_data, selected_ticker)
        st.plotly_chart(sentiment_chart, use_container_width=True)
        
        # News statistics
        if selected_ticker != 'ALL':
            ticker_news = news_data[news_data['ticker'] == selected_ticker]
        else:
            ticker_news = news_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Articles", len(ticker_news))
        with col2:
            avg_sentiment = ticker_news['sentiment_score'].mean()
            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
        with col3:
            positive_ratio = (ticker_news['sentiment_score'] > 0).mean()
            st.metric("Positive News %", f"{positive_ratio:.1%}")
        with col4:
            articles_per_day = len(ticker_news) / ticker_news['publishedAt'].dt.date.nunique()
            st.metric("Articles/Day", f"{articles_per_day:.1f}")
        
        # Recent news sample
        st.subheader("Recent News Sample")
        recent_news = ticker_news.nlargest(5, 'publishedAt')[['publishedAt', 'text', 'sentiment_score']]
        st.dataframe(recent_news, use_container_width=True)
    
    elif page == "Model Performance":
        st.header("🤖 Model Performance Analysis")
        
        if not model_results:
            st.warning("No model results available. Please run training first.")
            st.code("python train_advanced.py")
            return
        
        # Model comparison chart
        comparison_chart = create_model_comparison_chart(model_results)
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Model Results")
        
        results_df = pd.DataFrame([
            {
                'Model': model_name.replace('_', ' ').title(),
                'Validation Loss': results['best_val_loss'],
                'Validation Accuracy': f"{results['final_val_acc']:.3f}",
                'Parameters': f"{results['param_count']:,}",
                'Score': f"{results['final_val_acc'] / results['best_val_loss']:.2f}"
            }
            for model_name, results in model_results.items()
        ])
        
        results_df = results_df.sort_values('Score', ascending=False)
        st.dataframe(results_df, use_container_width=True)
    
    elif page == "Live Prediction":
        st.header("🔮 Live Market Prediction")
        
        # Check for trained models
        model_files = [f for f in os.listdir('experiments/financial_models_comparison/checkpoints/') 
                      if f.endswith('.pth')] if os.path.exists('experiments/financial_models_comparison/checkpoints/') else []
        
        if not model_files:
            st.warning("No trained models found. Please run training first.")
            st.code("python train_advanced.py")
            return
        
        # Model selection
        selected_model = st.selectbox("Select Model", model_files)
        
        # Ticker selection
        tickers = sorted(stock_data['Ticker'].unique()) if not stock_data.empty else []
        selected_ticker = st.selectbox("Select Ticker for Prediction", tickers)
        
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Generating prediction..."):
                # This would require implementing prediction logic
                # For now, show placeholder
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Simulated prediction
                    pred_return = np.random.normal(0, 0.02)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Predicted Return", f"{pred_return:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    pred_volatility = np.random.uniform(0.01, 0.05)
                    st.markdown('<div class="metric-card warning-card">', unsafe_allow_html=True)
                    st.metric("Predicted Volatility", f"{pred_volatility:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    confidence = np.random.uniform(0.6, 0.9)
                    color = "success-card" if confidence > 0.8 else "warning-card" if confidence > 0.7 else "danger-card"
                    st.markdown(f'<div class="metric-card {color}">', unsafe_allow_html=True)
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.success("Prediction generated successfully!")
                st.info("Note: This is a demo prediction. Implement actual model inference for real predictions.")

if __name__ == "__main__":
    main()