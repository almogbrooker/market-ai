#!/usr/bin/env python3
"""
Advanced training script using state-of-the-art architectures for financial markets
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import seaborn as sns
import logging
import os
import sys
from datetime import datetime
from tqdm import tqdm
import json
import glob

# Import our modules
from advanced_models import (
    create_advanced_model, FinancialLoss, 
    AdvancedLSTM, FinancialTransformer, EnsembleModel, WaveNet
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """Advanced trainer with comprehensive evaluation and model comparison"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"advanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_data = {}
        
        # Create experiment directory
        self.exp_dir = f"experiments/{self.experiment_name}"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f"{self.exp_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/plots", exist_ok=True)
        
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Experiment directory: {self.exp_dir}")
        
        # Auto-detect existing checkpoints
        self._auto_detect_checkpoints()
    
    def _auto_detect_checkpoints(self):
        """Auto-detect and load the latest checkpoints for each model"""
        logger.info("Checking for existing checkpoints...")
        
        checkpoint_dir = f"{self.exp_dir}/checkpoints"
        if not os.path.exists(checkpoint_dir):
            logger.info("No checkpoint directory found, starting fresh")
            return
        
        # Find all checkpoint files
        checkpoint_files = glob.glob(f"{checkpoint_dir}/*.pth")
        
        if not checkpoint_files:
            logger.info("No checkpoints found, starting fresh training")
            return
        
        # Group checkpoints by model name
        model_checkpoints = {}
        for checkpoint_path in checkpoint_files:
            filename = os.path.basename(checkpoint_path)
            
            # Extract model name from filename patterns:
            # best_model_name.pth or checkpoint_batch_N_model_name.pth
            if filename.startswith('best_'):
                model_name = filename[5:-4]  # Remove 'best_' and '.pth'
                checkpoint_type = 'best'
            elif filename.startswith('checkpoint_batch_'):
                parts = filename.split('_')
                if len(parts) >= 4:
                    model_name = '_'.join(parts[3:])[:-4]  # Remove '.pth'
                    batch_num = int(parts[2])
                    checkpoint_type = 'batch'
                else:
                    continue
            else:
                continue
            
            if model_name not in model_checkpoints:
                model_checkpoints[model_name] = []
            
            model_checkpoints[model_name].append({
                'path': checkpoint_path,
                'type': checkpoint_type,
                'batch_num': batch_num if checkpoint_type == 'batch' else 0,
                'mtime': os.path.getmtime(checkpoint_path)
            })
        
        # Select latest checkpoint for each model
        for model_name, checkpoints in model_checkpoints.items():
            # Prefer batch checkpoints over best checkpoints for resuming
            batch_checkpoints = [c for c in checkpoints if c['type'] == 'batch']
            
            if batch_checkpoints:
                # Get the latest batch checkpoint
                latest_checkpoint = max(batch_checkpoints, key=lambda x: (x['batch_num'], x['mtime']))
            else:
                # Fall back to best checkpoint
                best_checkpoints = [c for c in checkpoints if c['type'] == 'best']
                if best_checkpoints:
                    latest_checkpoint = max(best_checkpoints, key=lambda x: x['mtime'])
                else:
                    continue
            
            # Load checkpoint
            try:
                checkpoint_data = torch.load(latest_checkpoint['path'], map_location=self.device)
                self.checkpoint_data[model_name] = {
                    'data': checkpoint_data,
                    'path': latest_checkpoint['path']
                }
                
                logger.info(f"✓ Found checkpoint for {model_name}: {os.path.basename(latest_checkpoint['path'])}")
                logger.info(f"  - Epoch: {checkpoint_data.get('epoch', 'unknown')}")
                logger.info(f"  - Batch Counter: {checkpoint_data.get('batch_counter', 'unknown')}")
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint for {model_name}: {e}")
        
        if self.checkpoint_data:
            logger.info(f"Loaded {len(self.checkpoint_data)} model checkpoints")
        else:
            logger.info("No valid checkpoints found, starting fresh training")
    
    def load_and_prepare_data(self):
        """Load and prepare the training data"""
        logger.info("Loading training data...")
        
        # Load processed data
        if not os.path.exists('data/training_data.csv'):
            logger.error("Training data not found. Please run data preparation first.")
            raise FileNotFoundError("data/training_data.csv not found")
        
        df = pd.read_csv('data/training_data.csv')
        logger.info(f"Loaded {len(df)} training samples")
        
        # Feature columns (exclude non-numeric)
        exclude_cols = ['Date', 'Ticker']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create sequences by ticker
        ticker_data = {}
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].sort_values('Date')
            ticker_df = ticker_df.fillna(0)
            
            features = ticker_df[feature_cols].values
            close_prices = ticker_df['Close'].values
            
            # Calculate returns
            returns = np.zeros_like(close_prices)
            returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
            
            ticker_data[ticker] = {
                'features': features,
                'returns': returns,
                'dates': ticker_df['Date'].values
            }
        
        return ticker_data, feature_cols
    
    def create_sequences(self, ticker_data, feature_cols, sequence_length=30):
        """Create training sequences with longer lookback"""
        logger.info(f"Creating sequences with length {sequence_length}...")
        
        all_sequences = []
        all_targets = []
        all_volatility = []
        
        for ticker, data in ticker_data.items():
            features = data['features']
            returns = data['returns']
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                seq = features[i:i+sequence_length]
                target = returns[i+sequence_length]
                
                # Calculate volatility (rolling std of returns)
                vol_window = returns[max(0, i+sequence_length-10):i+sequence_length]
                volatility = np.std(vol_window) if len(vol_window) > 1 else 0.02
                
                all_sequences.append(seq)
                all_targets.append(target)
                all_volatility.append(volatility)
        
        logger.info(f"Created {len(all_sequences)} sequences")
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(all_sequences)).to(self.device)
        y = torch.FloatTensor(np.array(all_targets)).unsqueeze(1).to(self.device)
        vol = torch.FloatTensor(np.array(all_volatility)).unsqueeze(1).to(self.device)
        
        return X, y, vol, len(feature_cols)
    
    def train_model(self, model, train_loader, val_loader, model_name, epochs=100):
        """Train a single model with comprehensive monitoring"""
        logger.info(f"Training {model_name}...")
        
        # Advanced optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduling
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # Loss function
        criterion = FinancialLoss(return_weight=1.0, directional_weight=0.3, volatility_weight=0.2)
        
        # Training tracking
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        batch_counter = 0
        save_frequency = 100  # Save every 100 batches
        start_epoch = 0
        
        # Resume from checkpoint if available
        if model_name in self.checkpoint_data:
            logger.info(f"Resuming {model_name} from checkpoint...")
            try:
                checkpoint = self.checkpoint_data[model_name]['data']
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                batch_counter = checkpoint.get('batch_counter', 0)
                
                # Load training history if available
                if 'train_losses' in checkpoint:
                    train_losses = checkpoint['train_losses']
                    val_losses = checkpoint['val_losses']
                    train_accuracies = checkpoint['train_accuracies']
                    val_accuracies = checkpoint['val_accuracies']
                    best_val_loss = min(val_losses) if val_losses else float('inf')
                
                logger.info(f"✓ Resumed from epoch {start_epoch}, batch counter {batch_counter}")
                logger.info(f"  Previous best val loss: {best_val_loss:.6f}")
            except Exception as e:
                logger.warning(f"Failed to resume from checkpoint: {e}")
                logger.info("Starting fresh training...")
                start_epoch = 0
                batch_counter = 0
        
        for epoch in range(start_epoch, epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            
            for batch_idx, (batch_X, batch_y, batch_vol) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, FinancialTransformer):
                    outputs = model(batch_X)
                    predictions = outputs['return_prediction']
                    volatility_pred = outputs['volatility_prediction']
                    loss = criterion(predictions, batch_y, volatility_pred)
                elif isinstance(model, EnsembleModel):
                    outputs = model(batch_X)
                    predictions = outputs['meta_prediction']
                    loss = criterion(predictions, batch_y)
                else:
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_train_loss += loss.item()
                batch_counter += 1
                
                # Calculate accuracy
                pred_direction = torch.sign(predictions)
                true_direction = torch.sign(batch_y)
                accuracy = (pred_direction == true_direction).float().mean()
                epoch_train_acc += accuracy.item()
                
                # Save checkpoint every N batches
                if batch_counter % save_frequency == 0:
                    checkpoint_path = f"{self.exp_dir}/checkpoints/checkpoint_batch_{batch_counter}_{model_name}.pth"
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'batch': batch_idx,
                        'train_loss': loss.item(),
                        'batch_counter': batch_counter,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies
                    }
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"  Checkpoint saved at batch {batch_counter}: {checkpoint_path}")
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_train_acc = epoch_train_acc / len(train_loader)
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y, batch_vol in val_loader:
                    if isinstance(model, FinancialTransformer):
                        outputs = model(batch_X)
                        predictions = outputs['return_prediction']
                        volatility_pred = outputs['volatility_prediction']
                        loss = criterion(predictions, batch_y, volatility_pred)
                    elif isinstance(model, EnsembleModel):
                        outputs = model(batch_X)
                        predictions = outputs['meta_prediction']
                        loss = criterion(predictions, batch_y)
                    else:
                        predictions = model(batch_X)
                        loss = criterion(predictions, batch_y)
                    
                    epoch_val_loss += loss.item()
                    
                    pred_direction = torch.sign(predictions)
                    true_direction = torch.sign(batch_y)
                    accuracy = (pred_direction == true_direction).float().mean()
                    epoch_val_acc += accuracy.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            avg_val_acc = epoch_val_acc / len(val_loader)
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(avg_train_acc)
            val_accuracies.append(avg_val_acc)
            
            # Logging
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"{model_name} - Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            logger.info(f"  Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}")
            logger.info(f"  LR: {current_lr:.8f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_acc': avg_train_acc,
                    'val_acc': avg_val_acc
                }
                
                torch.save(checkpoint, f"{self.exp_dir}/checkpoints/best_{model_name}.pth")
                logger.info(f"  ★ New best {model_name}! Val Loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping for {model_name} at epoch {epoch+1}")
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss
        }
    
    def compare_models(self, X, y, vol, input_size):
        """Train and compare multiple advanced models"""
        logger.info("Starting model comparison...")
        
        # Train/validation split
        X_train, X_val, y_train, y_val, vol_train, vol_val = train_test_split(
            X, y, vol, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train, vol_train)
        val_dataset = TensorDataset(X_val, y_val, vol_val)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Define models to compare
        models_config = {
            'advanced_lstm_small': {
                'type': 'advanced_lstm',
                'params': {'num_layers': 4, 'hidden_size': 128, 'dropout': 0.2}
            },
            'advanced_lstm_large': {
                'type': 'advanced_lstm', 
                'params': {'num_layers': 6, 'hidden_size': 256, 'dropout': 0.3}
            },
            'financial_transformer_small': {
                'type': 'financial_transformer',
                'params': {'num_layers': 4, 'd_model': 256, 'n_heads': 8, 'dropout': 0.1}
            },
            'financial_transformer_large': {
                'type': 'financial_transformer',
                'params': {'num_layers': 8, 'd_model': 512, 'n_heads': 16, 'dropout': 0.1}
            },
            'ensemble_model': {
                'type': 'ensemble',
                'params': {'hidden_size': 256}
            },
            'wavenet_model': {
                'type': 'wavenet',
                'params': {'num_blocks': 4, 'layers_per_block': 12, 'residual_channels': 128}
            }
        }
        
        results = {}
        
        for model_name, config in models_config.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*60}")
            
            try:
                # Create model
                model = create_advanced_model(
                    config['type'], 
                    input_size, 
                    **config['params']
                ).to(self.device)
                
                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"Model parameters: {param_count:,}")
                
                # Train model
                model_results = self.train_model(
                    model, train_loader, val_loader, model_name, epochs=80
                )
                
                model_results['param_count'] = param_count
                model_results['config'] = config
                results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Save results
        with open(f"{self.exp_dir}/model_comparison.json", 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for k, v in results.items():
                json_results[k] = {
                    'best_val_loss': float(v['best_val_loss']),
                    'final_train_loss': float(v['train_losses'][-1]),
                    'final_val_loss': float(v['val_losses'][-1]),
                    'final_train_acc': float(v['train_accuracies'][-1]),
                    'final_val_acc': float(v['val_accuracies'][-1]),
                    'param_count': v['param_count'],
                    'config': v['config']
                }
            json.dump(json_results, f, indent=2)
        
        return results
    
    def plot_comparison(self, results):
        """Create comprehensive comparison plots"""
        logger.info("Creating comparison plots...")
        
        # Model comparison summary
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Best validation loss comparison
        model_names = list(results.keys())
        best_losses = [results[name]['best_val_loss'] for name in model_names]
        param_counts = [results[name]['param_count'] for name in model_names]
        
        axes[0, 0].bar(range(len(model_names)), best_losses)
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].set_title('Best Validation Loss by Model')
        axes[0, 0].set_ylabel('Validation Loss')
        
        # 2. Parameter count vs performance
        axes[0, 1].scatter(param_counts, best_losses)
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name, (param_counts[i], best_losses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Parameter Count')
        axes[0, 1].set_ylabel('Best Validation Loss')
        axes[0, 1].set_title('Model Efficiency (Parameters vs Performance)')
        axes[0, 1].set_xscale('log')
        
        # 3. Final accuracy comparison
        final_accs = [results[name]['val_accuracies'][-1] for name in model_names]
        axes[0, 2].bar(range(len(model_names)), final_accs)
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 2].set_title('Final Validation Accuracy')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random')
        axes[0, 2].legend()
        
        # 4. Training curves for best models
        best_3_models = sorted(results.items(), key=lambda x: x[1]['best_val_loss'])[:3]
        
        for i, (name, result) in enumerate(best_3_models):
            epochs = range(1, len(result['val_losses']) + 1)
            axes[1, 0].plot(epochs, result['train_losses'], label=f'{name} (Train)', alpha=0.7)
            axes[1, 0].plot(epochs, result['val_losses'], label=f'{name} (Val)', alpha=0.9)
        
        axes[1, 0].set_title('Training Curves - Top 3 Models')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
        # 5. Accuracy curves for best models
        for i, (name, result) in enumerate(best_3_models):
            epochs = range(1, len(result['val_accuracies']) + 1)
            axes[1, 1].plot(epochs, result['train_accuracies'], label=f'{name} (Train)', alpha=0.7)
            axes[1, 1].plot(epochs, result['val_accuracies'], label=f'{name} (Val)', alpha=0.9)
        
        axes[1, 1].set_title('Accuracy Curves - Top 3 Models')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        # 6. Model rankings
        rankings = []
        for name in model_names:
            score = (1 / results[name]['best_val_loss']) * results[name]['val_accuracies'][-1]
            rankings.append((name, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        names, scores = zip(*rankings)
        
        axes[1, 2].barh(range(len(names)), scores)
        axes[1, 2].set_yticks(range(len(names)))
        axes[1, 2].set_yticklabels(names)
        axes[1, 2].set_title('Overall Model Ranking\n(Accuracy / Loss)')
        axes[1, 2].set_xlabel('Score')
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/plots/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*80)
        
        for i, (name, score) in enumerate(rankings):
            result = results[name]
            logger.info(f"\n{i+1}. {name.upper()}")
            logger.info(f"   Best Val Loss: {result['best_val_loss']:.6f}")
            logger.info(f"   Final Val Acc: {result['val_accuracies'][-1]:.4f} ({result['val_accuracies'][-1]*100:.2f}%)")
            logger.info(f"   Parameters: {result['param_count']:,}")
            logger.info(f"   Score: {score:.4f}")
        
        # Find best model
        best_model = rankings[0][0]
        logger.info(f"\n🏆 BEST MODEL: {best_model.upper()}")
        logger.info(f"   Validation Loss: {results[best_model]['best_val_loss']:.6f}")
        logger.info(f"   Validation Accuracy: {results[best_model]['val_accuracies'][-1]:.4f}")
        logger.info(f"   Improvement over simple LSTM: {((results[best_model]['val_accuracies'][-1] - 0.5112) / 0.5112 * 100):+.2f}%")
    
    def run_experiment(self):
        """Run the complete advanced training experiment"""
        logger.info("Starting Advanced Training Experiment")
        logger.info("="*60)
        
        # Load data
        ticker_data, feature_cols = self.load_and_prepare_data()
        
        # Create sequences
        X, y, vol, input_size = self.create_sequences(ticker_data, feature_cols, sequence_length=30)
        
        logger.info(f"Dataset: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
        
        # Compare models
        results = self.compare_models(X, y, vol, input_size)
        
        # Create plots and analysis
        self.plot_comparison(results)
        
        logger.info("="*60)
        logger.info("Advanced Training Experiment Completed!")
        logger.info(f"Results saved to: {self.exp_dir}")
        logger.info("="*60)
        
        return results
    
    def check_models_status(self):
        """Check status and performance of existing models"""
        logger.info("Checking models status...")
        
        if not self.checkpoint_data:
            logger.info("No saved models found")
            return None
        
        # Read comparison results if available
        results_file = f"{self.exp_dir}/model_comparison.json"
        comparison_results = None
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    comparison_results = json.load(f)
                logger.info("Found model comparison results")
            except Exception as e:
                logger.warning(f"Error reading results: {e}")
        
        print("\n" + "="*80)
        print("📊 Models Status")
        print("="*80)
        
        model_info = []
        
        for model_name, checkpoint_info in self.checkpoint_data.items():
            checkpoint = checkpoint_info['data']
            path = checkpoint_info['path']
            
            # Basic model info
            epoch = checkpoint.get('epoch', 0)
            batch_counter = checkpoint.get('batch_counter', 0)
            train_loss = checkpoint.get('train_loss', 'N/A')
            
            # Detailed results if available
            detailed_results = None
            if comparison_results and model_name in comparison_results:
                detailed_results = comparison_results[model_name]
            
            model_info.append({
                'name': model_name,
                'epoch': epoch,
                'batch_counter': batch_counter,
                'train_loss': train_loss,
                'checkpoint_path': os.path.basename(path),
                'detailed_results': detailed_results
            })
        
        # Sort by performance if detailed data available
        if comparison_results:
            model_info.sort(key=lambda x: x['detailed_results']['best_val_loss'] if x['detailed_results'] else float('inf'))
        
        # Display information
        for i, info in enumerate(model_info):
            print(f"\n{i+1}. 🤖 {info['name'].upper()}")
            print(f"   📁 File: {info['checkpoint_path']}")
            print(f"   🔄 Epoch: {info['epoch']}")
            print(f"   📊 Batches: {info['batch_counter']:,}")
            
            if info['detailed_results']:
                results = info['detailed_results']
                print(f"   🎯 Validation Accuracy: {results['final_val_acc']:.4f} ({results['final_val_acc']*100:.2f}%)")
                print(f"   📉 Best Val Loss: {results['best_val_loss']:.6f}")
                print(f"   ⚙️  Parameters: {results['param_count']:,}")
                
                # Model quality assessment
                accuracy = results['final_val_acc']
                if accuracy > 0.55:
                    quality = "🟢 Excellent"
                elif accuracy > 0.52:
                    quality = "🟡 Good"
                else:
                    quality = "🔴 Needs Improvement"
                print(f"   📈 Quality: {quality}")
            else:
                print(f"   ⚠️  Last Loss: {info['train_loss']}")
                print(f"   📝 Status: Training in progress")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("💡 Recommendations:")
        
        if not comparison_results:
            print("• Training not yet completed - can continue from last checkpoint")
            print("• Run script again to continue training")
        else:
            best_model = model_info[0] if model_info else None
            if best_model and best_model['detailed_results']:
                best_acc = best_model['detailed_results']['final_val_acc']
                print(f"• Best model: {best_model['name']}")
                print(f"• Accuracy: {best_acc*100:.2f}%")
                
                if best_acc > 0.55:
                    print("• Model ready for production! 🚀")
                elif best_acc > 0.52:
                    print("• Model is decent - consider additional training or parameter tuning")
                else:
                    print("• Model needs improvement - try more data or different architecture")
        
        print("="*80)
        
        return model_info

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Financial Model Training')
    parser.add_argument('--experiment', type=str, default="financial_models_comparison", 
                       help='Experiment name')
    parser.add_argument('--check', action='store_true', help='Check existing models status only')
    args = parser.parse_args()
    
    trainer = AdvancedTrainer(args.experiment)
    
    if args.check:
        # Only check existing models
        return trainer.check_models_status()
    
    # Test the advanced models first
    logger.info("Testing advanced model architectures...")
    
    # Run the architecture test
    os.system("python advanced_models.py")
    
    # Run the advanced training experiment
    # The trainer will automatically detect and resume from checkpoints if they exist
    results = trainer.run_experiment()
    
    return results

if __name__ == "__main__":
    main()