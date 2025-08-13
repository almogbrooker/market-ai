#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import logging
import sys
import os
from pathlib import Path
from features_enhanced import add_fundamental_features
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced modules
from sentiment import analyze_news_sentiment
from features_enhanced import add_technical_indicators, merge_news_to_prices
from data_validation import validate_and_clean_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_training_data():
    """Prepare data for training using our enhanced pipeline"""
    logger.info("Preparing training data...")
    
    # Load the enhanced data
    try:
        prices_df = pd.read_csv('data/prices_enhanced.csv')
        news_df = pd.read_csv('data/news_enhanced.csv')
        
        logger.info(f"Loaded {len(prices_df)} price records and {len(news_df)} news articles")
        
        # Convert date columns
        prices_df['Date'] = pd.to_datetime(prices_df['Date'])
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
        
        # Analyze sentiment
        logger.info("Analyzing news sentiment...")
        try:
            news_df = analyze_news_sentiment(news_df, text_column='text')
            logger.info("Sentiment analysis completed")
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            # Create dummy sentiment scores
            news_df['sentiment_score'] = 0.0
        
        # Validate and clean data
        logger.info("Validating and cleaning data...")
        prices_clean, news_clean, validation_results = validate_and_clean_data(prices_df, news_df)
        
        # Add technical indicators
        logger.info("Adding technical indicators...")
        prices_enhanced = add_technical_indicators(prices_clean, validate_data=False)
        
        # Merge with news sentiment
        logger.info("Merging news sentiment with price data...")
        final_data = merge_news_to_prices(prices_enhanced, news_clean)
        
        logger.info(f"Final dataset shape: {final_data.shape}")
        logger.info(f"Columns: {final_data.columns.tolist()}")
        
        # Save processed data
        final_data.to_csv('data/training_data.csv', index=False)
        logger.info("Processed training data saved to data/training_data.csv")
        # Merge with news sentiment
        logger.info("Merging news sentiment with price data...")
        final_data = merge_news_to_prices(prices_enhanced, news_clean)

        # === NEW STEP: Add fundamental features ===
        logger.info("Adding fundamental features...")
        fundamentals_path = 'data/fundamentals.json'
        final_data = add_fundamental_features(final_data, fundamentals_path)
        # =========================================

        logger.info(f"Final dataset shape after adding fundamentals: {final_data.shape}")
        logger.info(f"Columns: {final_data.columns.tolist()}")

        # Save processed data
        final_data.to_csv('data/training_data.csv', index=False)
        logger.info("Processed training data saved to data/training_data.csv")

        return final_data
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        raise

def create_simple_torch_dataset(data_df):
    """Create a simple PyTorch dataset from the processed data"""
    
    # Feature columns (excluding non-numeric and target columns)
    exclude_cols = ['Date', 'Ticker']
    feature_cols = [col for col in data_df.columns if col not in exclude_cols]
    
    # Group by ticker to create sequences
    ticker_data = {}
    for ticker in data_df['Ticker'].unique():
        ticker_df = data_df[data_df['Ticker'] == ticker].sort_values('Date')
        
        # Fill any remaining NaN values
        ticker_df = ticker_df.fillna(0)
        
        # Extract features and create targets (next day returns)
        features = ticker_df[feature_cols].values
        
        # Calculate next day returns as targets
        close_prices = ticker_df['Close'].values
        returns = np.zeros_like(close_prices)
        returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
        
        ticker_data[ticker] = {
            'features': features,
            'targets': returns,
            'dates': ticker_df['Date'].values
        }
    
    return ticker_data, feature_cols

def train_simple_model():
    """Train a simple LSTM model on the processed data"""
    logger.info("Starting model training...")
    
    # Prepare data
    data_df = prepare_training_data()
    
    # Create torch dataset
    ticker_data, feature_cols = create_simple_torch_dataset(data_df)
    
    logger.info(f"Number of tickers: {len(ticker_data)}")
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # Simple LSTM model
    import torch.nn as nn
    
    class StockLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(StockLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # Use the last output
            output = self.fc(self.dropout(lstm_out[:, -1, :]))
            return output
    
    # Prepare training data
    all_sequences = []
    all_targets = []
    sequence_length = 20
    
    for ticker, data in ticker_data.items():
        features = data['features']
        targets = data['targets']
        
        # Create sequences
        for i in range(len(features) - sequence_length):
            seq = features[i:i+sequence_length]
            target = targets[i+sequence_length]
            
            all_sequences.append(seq)
            all_targets.append(target)
    
    logger.info(f"Created {len(all_sequences)} training sequences")
    
    if len(all_sequences) == 0:
        logger.error("No training sequences created!")
        return
    
    # Convert to tensors
    X = torch.FloatTensor(np.array(all_sequences))
    y = torch.FloatTensor(np.array(all_targets)).unsqueeze(1)
    
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Target data shape: {y.shape}")
    
    # Train/validation split
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_size = len(feature_cols)
    model = StockLSTM(input_size=input_size)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training parameters
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.6f}')
        logger.info(f'  Val Loss: {avg_val_loss:.6f}')
        logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save model
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'feature_cols': feature_cols,
                'input_size': input_size
            }, 'checkpoints/best_stock_model.pth')
            
            logger.info(f'  ★ New best model saved! Val Loss: {best_val_loss:.6f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info('Early stopping triggered!')
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info("Model saved to: checkpoints/best_stock_model.pth")
    
    # Quick evaluation
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).numpy().flatten()
        train_actuals = y_train.numpy().flatten()
        
        # Calculate metrics
        mse = np.mean((train_preds - train_actuals) ** 2)
        mae = np.mean(np.abs(train_preds - train_actuals))
        
        # Directional accuracy
        pred_direction = np.sign(train_preds)
        actual_direction = np.sign(train_actuals)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        logger.info("Training Set Evaluation:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.4f} ({directional_accuracy*100:.2f}%)")
    
    return model

def main():
    """Main training function"""
    logger.info("="*60)
    logger.info("Starting Enhanced Market AI Training")
    logger.info("="*60)
    
    try:
        # Train the model
        model = train_simple_model()
        
        logger.info("="*60)
        logger.info("Training completed successfully!")
        logger.info("="*60)
        
        return model
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()