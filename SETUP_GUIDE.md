# Financial AI Trading System - Setup Guide

## Quick Start

This guide will help you set up the complete financial AI trading system from scratch, including downloading all necessary data and training models.

## Prerequisites

- Python 3.8 or higher
- Git
- 8GB+ RAM recommended
- GPU (optional but recommended for training)

## 1. Clone and Setup

```bash
# Clone the repository
git clone git@github.com:almogbrooker/market-ai.git
cd market-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 2. Data Download and Preparation

### Step 1: Download Stock Price Data
```bash
# Fetch historical stock data for major tech companies
python data_fetch.py
```
This will create:
- `data/AAPL.csv`, `data/MSFT.csv`, etc. (individual stock files)
- `data/prices.csv` (combined price data)

### Step 2: Download News Data
```bash
# Fetch news data (this may take 10-15 minutes)
python fetch_news_once.py
```
This will create:
- `data/news.csv` (raw news data)
- `data/news.parquet` (optimized format)

### Step 3: Download Fundamental Data
```bash
# Fetch company fundamentals
python combine_news_data.py
```
This will create:
- `data/fundamentals.json` (company financial data)

### Step 4: Create Enhanced Features
```bash
# Generate technical indicators and sentiment scores
python features_enhanced.py
```
This will create:
- `data/prices_enhanced.csv` (prices with technical indicators)
- `data/news_enhanced.csv` (news with sentiment scores)

### Step 5: Prepare Training Data
```bash
# Combine all data into training format
python data_validation.py
```
This will create:
- `data/training_data.csv` (final training dataset)

## 3. Model Training

### Quick Training (Recommended)
```bash
# Train all models with automatic checkpointing
python train_advanced.py
```

### Custom Training
```bash
# Train with custom experiment name
python train_advanced.py --experiment my_trading_models

# Check existing models without training
python train_advanced.py --check
```

## 4. Expected File Structure After Setup

```
market-ai/
├── data/
│   ├── AAPL.csv              # Apple stock data
│   ├── MSFT.csv              # Microsoft stock data
│   ├── NVDA.csv              # NVIDIA stock data
│   ├── ... (other stocks)
│   ├── prices.csv            # Combined price data
│   ├── news.csv              # News articles
│   ├── news.parquet          # Optimized news data
│   ├── fundamentals.json     # Company fundamentals
│   ├── prices_enhanced.csv   # Prices with technical indicators
│   ├── news_enhanced.csv     # News with sentiment
│   └── training_data.csv     # Final training dataset
├── experiments/
│   └── financial_models_comparison/
│       ├── checkpoints/      # Model checkpoints
│       ├── plots/           # Performance charts
│       └── model_comparison.json
└── ... (Python files)
```

## 5. Data Size Expectations

- **Stock data**: ~50MB (10 stocks, 5 years)
- **News data**: ~500MB-1GB (depending on timeframe)
- **Training data**: ~100-200MB
- **Model checkpoints**: ~50-500MB per model
- **Total**: ~1-2GB

## 6. Troubleshooting

### Data Download Issues

**"No data downloaded"**
```bash
# Check internet connection and try again
python data_fetch.py
```

**"News API rate limit"**
```bash
# Wait a few minutes and retry
python fetch_news_once.py
```

**"Missing training data"**
```bash
# Run data preparation steps in order
python data_fetch.py
python fetch_news_once.py
python features_enhanced.py
python data_validation.py
```

### Training Issues

**"CUDA out of memory"**
- Close other applications
- Reduce batch size in `train_advanced.py`
- Use CPU training (slower but works)

**"Training stopped early"**
- Normal behavior if model converges
- Check results with `python train_advanced.py --check`

### Missing Dependencies
```bash
# Install additional packages if needed
pip install torch torchvision torchaudio
pip install transformers datasets
pip install yfinance requests beautifulsoup4
```

## 7. Configuration

### API Keys (Optional)
Create `config.py` with your API keys for enhanced data:
```python
NEWS_API_KEY = "your_news_api_key_here"
ALPHA_VANTAGE_KEY = "your_alpha_vantage_key_here"
```

### Stock Selection
Edit `config.py` to change which stocks to analyze:
```python
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM']
```

## 8. Usage After Setup

### Check Model Performance
```bash
python train_advanced.py --check
```

### Run Backtesting
```bash
python backtesting.py
```

### Start Web Interface
```bash
pip install -r requirements_ui.txt
python app.py
```

### Get Recent Predictions
```bash
python evaluate.py
```

## 9. Data Update Schedule

### Daily Updates
```bash
# Update recent prices and news
python fetch_recent_news.py
python data_fetch.py
```

### Weekly Full Refresh
```bash
# Full data refresh (recommended weekly)
python data_fetch.py
python fetch_news_once.py
python features_enhanced.py
python data_validation.py
```

## 10. Performance Optimization

### For Better Training Speed
- Use GPU: Install CUDA version of PyTorch
- Increase RAM: Close other applications
- Use SSD: Store data on fast storage

### For Better Predictions
- More data: Extend date ranges in `data_fetch.py`
- More stocks: Add tickers to `config.py`
- More features: Modify `features_enhanced.py`

## Support

If you encounter issues:
1. Check this guide first
2. Verify all data files exist in `data/` folder
3. Ensure virtual environment is activated
4. Check Python version compatibility

The system is designed to be robust and will automatically resume training from checkpoints if interrupted.

---

**Estimated Total Setup Time**: 30-60 minutes (depending on internet speed)

**Ready to Trade**: After setup completes, the system will have everything needed for AI-powered trading decisions!