#!/usr/bin/env python3
"""
Upgraded training script: time-aware splits, AMP, safe resume, optional causal masking
"""

import os
# Ensure allocator is configured *before* torch import
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import random
import json
import glob
import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging
import argparse
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.advanced_models import (
    create_advanced_model, FinancialLoss,
    AdvancedLSTM, FinancialTransformer, EnsembleModel, WaveNet, TSMixer, PatchTST, iTransformer
)
from data.alpha_loader import AlphaDataLoader, RankingLoss
from evaluation.conformal_prediction import ConformalPredictor, QuantileLoss, create_uncertainty_model
from training.sharpe_loss import create_risk_aware_loss

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
mlflow.start_run()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_causal_mask(batch_size: int, seq_len: int, device: torch.device):
    """Create a lower-triangular causal mask for self-attention.
    Shape: [1, 1, T, T] (broadcastable to [B, H, T, T]).
    """
    return torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class AdvancedTrainer:
    """Advanced trainer with comprehensive evaluation and model comparison"""
    
    def __init__(self, experiment_name: str = None, use_causal_mask: bool = True, zero_threshold_bps: float = 10.0,
                 label_mode: str = 'alpha', prediction_horizon: int = 1, neutral_zone_bps: float = 5.0):
        seed_everything(42)

        self.experiment_name = experiment_name or f"advanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_data = {}
        self.training_start_time = datetime.now()
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.use_causal_mask = use_causal_mask
        self.zero_threshold = zero_threshold_bps / 10_000.0  # convert bps to decimal
        
        # Alpha prediction configuration
        self.label_mode = label_mode
        self.prediction_horizon = prediction_horizon
        self.neutral_zone_bps = neutral_zone_bps
        
        # Initialize Alpha data loader
        self.alpha_loader = AlphaDataLoader(
            sequence_length=30,
            prediction_horizon=prediction_horizon,
            neutral_zone_bps=neutral_zone_bps,
            label_mode=label_mode
        )

        # Pre-create experiment folders
        self.exp_dir = f"experiments/{self.experiment_name}"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f"{self.exp_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/logs", exist_ok=True)

        # MLflow tracking
        mlflow.set_tag("mlflow.runName", self.experiment_name)
        mlflow.log_params({
            "use_causal_mask": use_causal_mask,
            "zero_threshold_bps": zero_threshold_bps,
            "label_mode": label_mode,
            "prediction_horizon": prediction_horizon,
            "neutral_zone_bps": neutral_zone_bps,
        })
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            mlflow.log_param("git_commit", commit_hash)
        except Exception as e:
            logger.warning(f"Could not retrieve git commit: {e}")
        mlflow.log_param("dataset_version", self._get_dataset_version())
        
        # Model configurations for manual training
        self.model_configs = {
            'advanced_lstm_small': {
                'type': 'advanced_lstm',
                'params': {'num_layers': 3, 'hidden_size': 96, 'dropout': 0.4}  # BALANCED: Moderate size cut + normal dropout
            },
            'advanced_lstm_large': {
                'type': 'advanced_lstm',
                'params': {'num_layers': 4, 'hidden_size': 192, 'dropout': 0.48}  # BALANCED: Proper dropout range (0.40-0.50)
            },
            'financial_transformer_small': {
                'type': 'financial_transformer',
                'params': {'num_layers': 3, 'd_model': 128, 'n_heads': 4, 'dropout': 0.25, 'drop_path': 0.05}  # ENHANCED: Added DropPath
            },
            'financial_transformer_large': {
                'type': 'financial_transformer',
                'params': {'num_layers': 8, 'd_model': 512, 'n_heads': 16, 'dropout': 0.3, 'drop_path': 0.1}  # ENHANCED: Added DropPath
            },
            'ensemble_model': {
                'type': 'ensemble',
                'params': {'hidden_size': 64}  # BALANCED: Halved size, will use normal regularization
            },
            'wavenet_model': {
                'type': 'wavenet',
                'params': {'num_blocks': 2, 'layers_per_block': 8, 'residual_channels': 64}  # BALANCED: Smaller model
            },
            'tsmixer_model': {
                'type': 'tsmixer',
                'params': {'seq_len': 30, 'hidden_dim': 256, 'num_blocks': 8, 'dropout': 0.15}  # TSMixer for comparison
            },
            'patchtst_model': {
                'type': 'patchtst',
                'params': {'seq_len': 30, 'patch_len': 8, 'stride': 4, 'd_model': 256, 'n_heads': 8, 'num_layers': 6, 'dropout': 0.1}  # PatchTST (NeurIPS 2023)
            },
            'itransformer_model': {
                'type': 'itransformer', 
                'params': {'seq_len': 30, 'd_model': 256, 'n_heads': 8, 'num_layers': 6, 'dropout': 0.1}  # iTransformer (ICML 2024)
            }
        }

        # Detailed file logging
        log_file = f"{self.exp_dir}/logs/training_session.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        logger.info(f"Detailed logging enabled: {log_file}")

        logger.info("="*80)
        logger.info("🤖 ADVANCED AI TRADING MODEL TRAINER - 12 HOUR SESSION")
        logger.info("="*80)
        logger.info(f"🗂️  Experiment: {self.experiment_name}")
        logger.info(f"💻 Device: {self.device}")
        logger.info(f"🕒 Session Start: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📁 Directory: {self.exp_dir}")
        logger.info(f"⏱️  Planned Duration: 12 hours")
        logger.info(f"🎯 Target: >60% validation accuracy")
        logger.info("="*80)

        # Attempt to resume training from checkpoints created in prior runs
        self._auto_detect_checkpoints()

    def _get_dataset_version(self):
        for env_var in ("DVC_DATA_VERSION", "LAKEFS_COMMIT", "DATASET_VERSION"):
            val = os.environ.get(env_var)
            if val:
                return val
        return "unknown"

    # ----------------------- Session Logging -----------------------
    def _log_session_status(self):
        current_time = datetime.now()
        elapsed = current_time - self.training_start_time
        hours = elapsed.total_seconds() / 3600
        logger.info("="*60)
        logger.info("📊 TRAINING SESSION STATUS")
        logger.info("="*60)
        logger.info(f"⏰ Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  Elapsed Time: {hours:.2f} hours ({elapsed})")
        logger.info(f"📈 Progress: {(hours/12)*100:.1f}% of 12-hour session")
        logger.info(f"⏳ Remaining: {max(0, 12-hours):.2f} hours")
        logger.info("="*60)

    # ----------------------- Checkpoints ---------------------------
    def _auto_detect_checkpoints(self):
        logger.info("Checking for existing checkpoints...")
        ckpt_dir = f"{self.exp_dir}/checkpoints"
        if not os.path.exists(ckpt_dir):
            logger.info("No checkpoint directory found, starting fresh")
            return

        files = glob.glob(f"{ckpt_dir}/best_*.pth") + glob.glob(f"{ckpt_dir}/latest_*.pth")
        if not files:
            logger.info("No checkpoints found, starting fresh training")
            return

        model_map = {}
        for path in files:
            filename = os.path.basename(path)
            if filename.startswith("best_") or filename.startswith("latest_"):
                model_name = filename.split("_", 1)[1][:-4]  # strip prefix & .pth
                model_map.setdefault(model_name, []).append(path)

        for model_name, paths in model_map.items():
            latest = max(paths, key=os.path.getmtime)
            try:
                data = torch.load(latest, map_location=self.device)
                self.checkpoint_data[model_name] = {"data": data, "path": latest}
                logger.info(f"✓ Found checkpoint for {model_name}: {os.path.basename(latest)}")
                logger.info(f"  - Epoch: {data.get('epoch','?')} | Global step: {data.get('global_step','?')}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint for {model_name}: {e}")

        if self.checkpoint_data:
            logger.info(f"Loaded {len(self.checkpoint_data)} model checkpoints")
        else:
            logger.info("No valid checkpoints found, starting fresh training")

    # ----------------------- Data Loading --------------------------
    def load_and_prepare_data_alpha(self):
        """Load data using Alpha data loader for cross-sectional prediction"""
        logger.info(f"Loading Alpha data (mode: {self.label_mode}, horizon: {self.prediction_horizon}D)...")
        
        X, y, metadata = self.alpha_loader.load_data()
        
        logger.info(f"Alpha dataset loaded: {X.shape[0]} samples")
        logger.info(f"Features: {X.shape[1]} timesteps x {X.shape[2]} features")
        logger.info(f"Target mode: {self.label_mode}")
        
        return X, y, metadata
        
    def load_and_prepare_data(self):
        logger.info("Loading training data...")
        data_files = [
            'data/training_data_with_financials.csv',
            'data/training_data_2020_2024_complete.csv',
            'data/training_data_with_social.csv',
            'data/training_data.csv'
        ]
        df = None
        used = None
        for f in data_files:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f)
                    used = f
                    logger.info(f"Loaded training data from: {f}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")
        if df is None:
            raise FileNotFoundError("No training data files found")

        # Basic checks
        if 'Date' not in df.columns or 'Ticker' not in df.columns or 'Close' not in df.columns:
            raise ValueError("Dataset must contain 'Date', 'Ticker', and 'Close' columns")

        # Ensure Date typed and sorted
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Ticker']).sort_values(['Ticker','Date']).reset_index(drop=True)

        # Choose feature columns (exclude id/date)
        exclude_cols = ['Date', 'Ticker', 'date']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Cast feature columns to numeric, fill NaNs
        for col in feature_cols[:]:
            s = pd.to_numeric(df[col], errors='coerce')
            if s.isna().all():
                logger.warning(f"Dropping non-numeric column: {col}")
                feature_cols.remove(col)
            else:
                df[col] = s.fillna(0.0)

        # Clean Close per ticker
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Close'] = df.groupby('Ticker')['Close'].transform(lambda s: s.ffill().bfill())
        # Drop rows still missing/invalid Close
        df = df[np.isfinite(df['Close']) & (df['Close'] > 0)].copy()

        # Normalize features per ticker (z-score, safe denom)
        df[feature_cols] = df.groupby('Ticker')[feature_cols].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )

        # Replace any remaining non-finite feature values (rare)
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Build per-ticker arrays with *log returns* (stable)
        ticker_data = {}
        tickers = df['Ticker'].unique()
        for t in tickers:
            tdf = df[df['Ticker'] == t].sort_values('Date').reset_index(drop=True)
            features = tdf[feature_cols].values.astype(np.float32)

            close = tdf['Close'].astype(np.float64).values
            # safe ratio & log; guard tiny/zero with clip
            prev = np.clip(close[:-1], 1e-12, None)
            ratio = np.clip(close[1:] / prev, 1e-12, 1e12)
            logret = np.zeros_like(close, dtype=np.float32)
            logret[1:] = np.log(ratio).astype(np.float32)

            # Final sanity: replace non-finite returns with 0
            logret[~np.isfinite(logret)] = 0.0

            ticker_data[t] = {
                'features': features,
                'returns': logret,
                'dates': tdf['Date'].values
            }

        # Quick analysis
        logger.info("="*80)
        logger.info("📊 COMPREHENSIVE DATASET ANALYSIS")
        logger.info("="*80)
        logger.info(f"📁 Data Source: {used}")
        logger.info(f"📈 Total Records: {len(df):,}")
        logger.info(f"📅 Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        logger.info(f"🏢 Companies: {len(tickers)} tickers")
        logger.info(f"📊 Tickers: {', '.join(sorted(tickers))}")
        return ticker_data, feature_cols

    # ----------------------- Sequence Builder ----------------------
    def create_sequences(self, ticker_data, feature_cols, sequence_length=30):
        """Create training sequences with a longer lookback (CPU tensors only), NaN/Inf safe."""
        logger.info(f"Creating sequences with length {sequence_length}...")
        all_seqs, all_tgts, all_vol, all_tickers = [], [], [], []

        n_dropped_seq = 0
        for ticker, data in ticker_data.items():
            X = data['features']          # [N, F]
            r = data['returns']           # [N]
            n = len(X)
            if n <= sequence_length:
                continue

            for i in range(n - sequence_length):
                seq = X[i:i+sequence_length]
                tgt = float(r[i+sequence_length])

                # Drop if any non-finite
                if not np.isfinite(tgt) or not np.isfinite(seq).all():
                    n_dropped_seq += 1
                    continue

                # Optional: clip target to robust range (e.g., +/- 20%)
                tgt = float(np.clip(tgt, -0.20, 0.20))

                # rolling std of returns as volatility proxy
                vw = r[max(0, i+sequence_length-10): i+sequence_length]
                vol = float(np.std(vw)) if len(vw) > 1 else 0.02
                if not np.isfinite(vol):
                    vol = 0.02

                all_seqs.append(seq.astype(np.float32))
                all_tgts.append(tgt)
                all_vol.append(vol)
                all_tickers.append(ticker)

        logger.info(f"Created {len(all_seqs)} sequences (dropped {n_dropped_seq} bad sequences)")

        # CPU tensors; move to GPU per-batch during training
        X   = torch.tensor(np.array(all_seqs, dtype=np.float32))
        y   = torch.tensor(np.array(all_tgts, dtype=np.float32)).unsqueeze(1)
        vol = torch.tensor(np.array(all_vol,  dtype=np.float32)).unsqueeze(1)

        return X, y, vol, len(feature_cols), all_tickers
        # ----------------------- Time-aware split ----------------------
    def _time_split_by_ticker(self, X, y, vol, ticker_ids, val_frac=0.2):
        """Chronological split per ticker to avoid overlapping-window leakage."""
        X_tr, y_tr, v_tr = [], [], []
        X_va, y_va, v_va = [], [], []

        tickers = sorted(set(ticker_ids))
        for t in tickers:
            idx = [i for i, tt in enumerate(ticker_ids) if tt == t]
            if len(idx) < 5:
                continue
            split = int(len(idx) * (1 - val_frac))
            tri, vai = idx[:split], idx[split:]
            if tri and vai:
                X_tr.append(X[tri]); y_tr.append(y[tri]); v_tr.append(vol[tri])
                X_va.append(X[vai]); y_va.append(y[vai]); v_va.append(vol[vai])

        X_train = torch.cat(X_tr, dim=0) if X_tr else X[:0]
        y_train = torch.cat(y_tr, dim=0) if y_tr else y[:0]
        v_train = torch.cat(v_tr, dim=0) if v_tr else vol[:0]

        X_val = torch.cat(X_va, dim=0) if X_va else X[:0]
        y_val = torch.cat(y_va, dim=0) if y_va else y[:0]
        v_val = torch.cat(v_va, dim=0) if v_va else vol[:0]
        return X_train, X_val, y_train, y_val, v_train, v_val

                # ----------------------- Training ------------------------------
    def train_model(self, model, train_loader, val_loader, model_name, epochs=2000):
        """Train a single model with stability guards (AMP off for LSTM, NaN watchdog, safe checkpoints)."""
        import copy
        
        # Clear GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"🧹 Cleared GPU memory before training {model_name}")

        start_time = datetime.now()
        logger.info("="*80)
        logger.info(f"🚀 STARTING TRAINING: {model_name.upper()}")
        logger.info("="*80)
        logger.info(f"💻 Device: {self.device} | AMP: {self.use_amp}")
        logger.info(f"🎯 Target Epochs: {epochs}")
        logger.info(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"🔄 Batches/Epoch: {len(train_loader)} | Val Batches: {len(val_loader)}")
        self._log_session_status()

        # --- Safer defaults for LSTM (turn off AMP, lower LR) ---
        is_lstm = isinstance(model, AdvancedLSTM)
        use_amp_local = (self.use_amp and (not is_lstm))
        # BALANCED ANTI-OVERFITTING: Normal regularization scales for healthy training
        is_large_transformer = isinstance(model, FinancialTransformer) and model.d_model > 200
        is_ensemble = isinstance(model, EnsembleModel)
        is_wavenet = isinstance(model, WaveNet)
        
        # STRONGER anti-overfitting: Higher weight decay + better early stopping
        max_lr = 8e-4  # Slightly lower max LR for all models
        weight_decay = 8e-4  # Much higher weight decay to prevent overfitting
        
        # Special case for large transformer (keep stable)
        if is_large_transformer:
            max_lr = 6e-4  # Even lower for stability
            
        base_lr = max_lr / 10.0  # OneCycle starts at 1/10 of max_lr

        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.999))

        steps_per_epoch = max(1, len(train_loader))
        scheduler_cfg = dict(
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )

        # Choose loss function based on label mode
        if self.label_mode == 'alpha':
            criterion = FinancialLoss(return_weight=1.0, directional_weight=0.3, volatility_weight=0.2)
        elif self.label_mode == 'cls':
            criterion = nn.CrossEntropyLoss()
        elif self.label_mode == 'rank':
            criterion = RankingLoss(temperature=1.0)
        else:
            raise ValueError(f"Unknown label mode: {self.label_mode}")

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        best_val_loss = float('inf')
        patience, patience_counter = 100, 0  # Proper patience for OneCycle (accounts for warm-up + peak noise)

        global_step = 0
        start_epoch = 0

        # ---------- Safe resume (but don't keep bad states) ----------
        if model_name in self.checkpoint_data:
            logger.info(f"Resuming {model_name} from checkpoint...")
            try:
                ckpt = self.checkpoint_data[model_name]['data']

                # Before loading, sanity-check the state dict tensors are finite
                bad = False
                for k, v in ckpt['model_state_dict'].items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        bad = True
                        break
                if bad:
                    logger.warning("⚠️  Checkpoint contains non-finite weights. Ignoring resume for safety.")
                else:
                    model.load_state_dict(ckpt['model_state_dict'])
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    start_epoch = int(ckpt.get('epoch', 0))
                    global_step = int(ckpt.get('global_step', 0))
                    train_losses = ckpt.get('train_losses', [])
                    val_losses = ckpt.get('val_losses', [])
                    train_accuracies = ckpt.get('train_accuracies', [])
                    val_accuracies = ckpt.get('val_accuracies', [])
                    if val_losses:
                        best_val_loss = min(val_losses)
                    logger.info(f"✓ Resumed from epoch {start_epoch}, global_step {global_step}")
            except Exception as e:
                logger.warning(f"Failed to resume {model_name}: {e}")
                start_epoch, global_step = 0, 0

        # Create scheduler aligned to current global_step, without stepping it
        last_epoch = global_step - 1 if global_step > 0 else -1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, last_epoch=last_epoch, **scheduler_cfg)

        # metrics CSV
        metrics_csv = f"{self.exp_dir}/logs/{model_name}_metrics.csv"
        if not os.path.exists(metrics_csv):
            with open(metrics_csv, "w") as f:
                f.write("epoch,train_loss,val_loss,train_acc,val_acc,lr\n")

        # Keep a last-good copy of weights for rollback
        last_good_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # ---------- Training loop ----------
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_train_loss, epoch_train_acc = 0.0, 0.0
            num_train_batches = 0

            for batch_X, batch_y, batch_vol in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                optimizer.zero_grad(set_to_none=True)

                # Move batch to device
                batch_X   = batch_X.to(self.device, non_blocking=True)
                batch_y   = batch_y.to(self.device, non_blocking=True)
                batch_vol = batch_vol.to(self.device, non_blocking=True)

                # Causal mask (Transformer only)
                causal_mask = None
                if self.use_causal_mask and isinstance(model, FinancialTransformer):
                    bsz, seq_len, _ = batch_X.shape
                    causal_mask = build_causal_mask(bsz, seq_len, self.device)

                # Forward + loss
                with torch.amp.autocast("cuda", enabled=use_amp_local):
                    if isinstance(model, FinancialTransformer):
                        outs = model(batch_X, causal_mask=causal_mask)
                        pred = outs['return_prediction']
                        vol_pred = outs.get('volatility_prediction', None)
                        loss = criterion(pred, batch_y, vol_pred)  # Use vol_pred if available
                    elif isinstance(model, EnsembleModel):
                        outs = model(batch_X)
                        pred = outs['meta_prediction']
                        vol_pred = outs.get('volatility', None)  # Ensemble has volatility output
                        loss = criterion(pred, batch_y, vol_pred)  # Use batch_vol for all models
                    else:
                        # LSTM/WaveNet now return uncertainty dictionaries
                        outs = model(batch_X)
                        pred = outs['return_prediction']
                        vol_pred = outs.get('volatility_prediction', None)  # Use model's uncertainty output
                        loss = criterion(pred, batch_y, vol_pred)  # Use model's own volatility prediction

                # Skip bad batches
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss detected — skipping this batch")
                    self.scaler.update()
                    continue

                # Backward
                prev_step_count = getattr(optimizer, "_step_count", 0)
                self.scaler.scale(loss).backward()

                # Extra safety: clip by value & norm
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                self.scaler.step(optimizer)
                self.scaler.update()

                # Only step scheduler if optimizer stepped
                if getattr(optimizer, "_step_count", 0) > prev_step_count:
                    scheduler.step()

                # Watchdog: ensure parameters remain finite; if not, rollback + shrink LR
                bad_params = False
                with torch.no_grad():
                    for p in model.parameters():
                        if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                            bad_params = True
                            break
                if bad_params:
                    logger.warning("⚠️  Non-finite weights detected — restoring last good state and reducing LR by 50%")
                    model.load_state_dict({k: v.to(self.device) for k, v in last_good_state.items()})
                    for g in optimizer.param_groups:
                        g['lr'] = max(g['lr'] * 0.5, 1e-6)
                    continue
                else:
                    # Update last good state occasionally (every 50 steps)
                    if (global_step % 50) == 0:
                        last_good_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                # Metrics
                epoch_train_loss += float(loss.detach().cpu())
                truth = batch_y.detach()
                preds = pred.detach()
                mask = torch.abs(truth) >= self.zero_threshold
                if mask.any():
                    acc = (torch.sign(preds[mask]) == torch.sign(truth[mask])).float().mean().item()
                    epoch_train_acc += acc
                num_train_batches += 1
                global_step += 1

                # Periodic checkpoint (only if finite)
                if global_step % 100 == 0:
                    ckpt_path = f"{self.exp_dir}/checkpoints/latest_{model_name}.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies
                    }, ckpt_path)
                    logger.info(f"  Latest checkpoint saved: {ckpt_path}")

            # Averages
            avg_train_loss = (epoch_train_loss / max(1, num_train_batches))
            avg_train_acc  = (epoch_train_acc  / max(1, num_train_batches))

            # ---------- Validation ----------
            model.eval()
            epoch_val_loss, epoch_val_acc = 0.0, 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch_X, batch_y, batch_vol in val_loader:
                    batch_X   = batch_X.to(self.device, non_blocking=True)
                    batch_y   = batch_y.to(self.device, non_blocking=True)
                    batch_vol = batch_vol.to(self.device, non_blocking=True)

                    causal_mask = None
                    if self.use_causal_mask and isinstance(model, FinancialTransformer):
                        bsz, seq_len, _ = batch_X.shape
                        causal_mask = build_causal_mask(bsz, seq_len, self.device)

                    with torch.amp.autocast("cuda", enabled=use_amp_local):
                        if isinstance(model, FinancialTransformer):
                            outs = model(batch_X, causal_mask=causal_mask)
                            pred = outs['return_prediction']
                            vol_pred = outs.get('volatility_prediction', None)
                            loss = criterion(pred, batch_y, vol_pred)  # Use vol_pred if available
                        elif isinstance(model, EnsembleModel):
                            outs = model(batch_X)
                            pred = outs['meta_prediction']
                            vol_pred = outs.get('volatility', None)  # Ensemble has volatility output
                            loss = criterion(pred, batch_y, vol_pred)  # Use volatility for all models
                        else:
                            # LSTM/WaveNet now return uncertainty dictionaries
                            outs = model(batch_X)
                            pred = outs['return_prediction']
                            vol_pred = outs.get('volatility_prediction', None)  # Use model's uncertainty output
                            loss = criterion(pred, batch_y, vol_pred)  # Use model's own volatility prediction

                    if torch.isfinite(loss):
                        epoch_val_loss += float(loss.detach().cpu())
                        mask = torch.abs(batch_y) >= self.zero_threshold
                        if mask.any():
                            acc = (torch.sign(pred) == torch.sign(batch_y))[mask].float().mean().item()
                            epoch_val_acc += acc
                        num_val_batches += 1

            avg_val_loss = epoch_val_loss / max(1, num_val_batches)
            avg_val_acc  = epoch_val_acc  / max(1, num_val_batches)

            train_losses.append(float(avg_train_loss))
            val_losses.append(float(avg_val_loss))
            train_accuracies.append(float(avg_train_acc))
            val_accuracies.append(float(avg_val_acc))

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": avg_train_acc,
                "val_acc": avg_val_acc
            }, step=epoch + 1)

            # Progress logs
            if epoch % 10 == 0 or epoch == epochs - 1:
                now = datetime.now()
                elapsed_h = (now - start_time).total_seconds() / 3600
                denom = max(1, epoch - start_epoch + 1)
                total_est = elapsed_h / denom * epochs
                remaining_h = max(0, total_est - elapsed_h)
                try:
                    current_lr = scheduler.get_last_lr()[0]
                except Exception:
                    current_lr = optimizer.param_groups[0]['lr']
                logger.info("="*60)
                logger.info(f"📊 {model_name.upper()} - EPOCH {epoch+1}/{epochs}")
                logger.info("="*60)
                logger.info(f"⏰ Now: {now.strftime('%H:%M:%S')}")
                logger.info(f"⏱️  Elapsed: {elapsed_h:.2f}h | Remaining: {remaining_h:.2f}h")
                logger.info(f"🎯 Train Loss: {avg_train_loss:.6f} → Val Loss: {avg_val_loss:.6f}")
                logger.info(f"📈 Train Acc: {avg_train_acc:.4f} → Val Acc: {avg_val_acc:.4f}")
                logger.info(f"🔧 LR: {current_lr:.2e} | Best Val Loss: {best_val_loss:.6f}")
                logger.info(f"⏳ Patience: {patience_counter}/{patience}")
                # Memory monitoring
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    gpu_cached = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"🔧 GPU Memory: {gpu_mem:.2f}GB allocated, {gpu_cached:.2f}GB cached")

            # Best checkpoint (only if finite)
            if math.isfinite(avg_val_loss) and (avg_val_loss < best_val_loss):
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'train_loss': float(avg_train_loss),
                    'val_loss': float(avg_val_loss),
                    'train_acc': float(avg_train_acc),
                    'val_acc': float(avg_val_acc)
                }, f"{self.exp_dir}/checkpoints/best_{model_name}.pth")
                logger.info(f"  ★ New best {model_name}! Val Loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1

            # Metrics CSV
            with open(metrics_csv, "a") as f:
                try:
                    lr = scheduler.get_last_lr()[0]
                except Exception:
                    lr = optimizer.param_groups[0]['lr']
                f.write(f"{epoch+1},{avg_train_loss},{avg_val_loss},{avg_train_acc},{avg_val_acc},{lr}\n")

            if patience_counter >= patience:
                logger.info(f"Early stopping for {model_name} at epoch {epoch+1}")
                break
        try:
            mlflow.log_artifact(metrics_csv, artifact_path=model_name)
            best_ckpt = f"{self.exp_dir}/checkpoints/best_{model_name}.pth"
            if os.path.exists(best_ckpt):
                mlflow.log_artifact(best_ckpt, artifact_path=model_name)
        except Exception as e:
            logger.warning(f"MLflow artifact logging failed: {e}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss
        }
    # ----------------------- Comparison ----------------------------
    def compare_models(self, X, y, vol, input_size, ticker_ids):
        """Train and compare multiple advanced models (time-aware split, worker-safe loaders)."""
        logger.info("Starting model comparison...")

        # time-aware split (no leakage)
        X_train, X_val, y_train, y_val, vol_train, vol_val = self._time_split_by_ticker(
            X, y, vol, ticker_ids, val_frac=0.2
        )

        train_dataset = TensorDataset(X_train, y_train, vol_train)
        val_dataset   = TensorDataset(X_val,   y_val,   vol_val)

        batch_size = 8 if torch.cuda.is_available() else 16  # Reduced for stability
        pin_mem = torch.cuda.is_available()

        # *** workers=0 + persistent_workers=False (safe with CPU tensors) ***
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin_mem, persistent_workers=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=pin_mem, persistent_workers=False
        )

        models_config = {
            'advanced_lstm_small': {
                'type': 'advanced_lstm',
                'params': {'num_layers': 4, 'hidden_size': 64, 'dropout': 0.4}  # ANTI-OVERFITTING: Tiny model + optimal dropout
            },
            'advanced_lstm_large': {
                'type': 'advanced_lstm',
                'params': {'num_layers': 6, 'hidden_size': 128, 'dropout': 0.45}  # ANTI-OVERFITTING: Smaller + proper dropout
            },
            'financial_transformer_small': {
                'type': 'financial_transformer',
                'params': {'num_layers': 4, 'd_model': 128, 'n_heads': 4, 'dropout': 0.2}  # ANTI-OVERFITTING: Much smaller + optimal dropout
            },
            'financial_transformer_large': {
                'type': 'financial_transformer',
                'params': {'num_layers': 6, 'd_model': 256, 'n_heads': 8, 'dropout': 0.3}  # ANTI-OVERFITTING: Smaller + proper dropout
            },
            'ensemble_model': {
                'type': 'ensemble',
                'params': {'hidden_size': 128}  # ANTI-OVERFITTING: Smaller ensemble
            },
            'wavenet_model': {
                'type': 'wavenet',
                'params': {'num_blocks': 4, 'layers_per_block': 12, 'residual_channels': 128}
            }
        }

        results = {}

        for i, (model_name, cfg) in enumerate(models_config.items(), 1):
            logger.info("="*80)
            logger.info(f"🤖 MODEL {i}/{len(models_config)}: {model_name.upper()}")
            logger.info("="*80)
            logger.info(f"🏗️  Architecture: {cfg['type']}")
            logger.info(f"⚙️  Parameters: {cfg['params']}")
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_param("model_type", cfg['type'])
                mlflow.log_params(cfg['params'])
                self._log_session_status()

                try:
                    model = create_advanced_model(cfg['type'], input_size, **cfg['params']).to(self.device)
                    params = sum(p.numel() for p in model.parameters())
                    logger.info(f"Model parameters: {params:,}")

                    model_results = self.train_model(model, train_loader, val_loader, model_name, epochs=2000)
                    model_results['param_count'] = params
                    model_results['config'] = cfg
                    mlflow.log_metric("best_val_loss", model_results['best_val_loss'])
                    if model_results['val_accuracies']:
                        mlflow.log_metric("final_val_acc", model_results['val_accuracies'][-1])
                    results[model_name] = model_results
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    continue

        # Save comparison summary
        out_json = f"{self.exp_dir}/model_comparison.json"
        with open(out_json, "w") as f:
            json.dump({
                k: {
                    'best_val_loss': float(v['best_val_loss']),
                    'final_train_loss': float(v['train_losses'][-1]),
                    'final_val_loss': float(v['val_losses'][-1]),
                    'final_train_acc': float(v['train_accuracies'][-1]),
                    'final_val_acc': float(v['val_accuracies'][-1]),
                    'param_count': v['param_count'],
                    'config': v['config']
                }
                for k, v in results.items()
            }, f, indent=2)

        return results

    # ----------------------- Plotting ------------------------------
    def plot_comparison(self, results):
        logger.info("Creating comparison plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        model_names = list(results.keys())
        best_losses = [results[n]['best_val_loss'] for n in model_names]
        param_counts = [results[n]['param_count'] for n in model_names]

        axes[0,0].bar(range(len(model_names)), best_losses)
        axes[0,0].set_xticks(range(len(model_names))); axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,0].set_title('Best Validation Loss by Model'); axes[0,0].set_ylabel('Validation Loss')

        axes[0,1].scatter(param_counts, best_losses)
        for i, n in enumerate(model_names):
            axes[0,1].annotate(n, (param_counts[i], best_losses[i]), xytext=(5,5), textcoords='offset points', fontsize=8)
        axes[0,1].set_xlabel('Parameter Count'); axes[0,1].set_ylabel('Best Validation Loss')
        axes[0,1].set_title('Model Efficiency (Parameters vs Performance)'); axes[0,1].set_xscale('log')

        final_accs = [results[n]['val_accuracies'][-1] for n in model_names]
        axes[0,2].bar(range(len(model_names)), final_accs)
        axes[0,2].set_xticks(range(len(model_names))); axes[0,2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,2].set_title('Final Validation Accuracy'); axes[0,2].set_ylabel('Accuracy')
        axes[0,2].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random'); axes[0,2].legend()

        best_3 = sorted(results.items(), key=lambda x: x[1]['best_val_loss'])[:3]
        for name, res in best_3:
            epochs = range(1, len(res['val_losses'])+1)
            axes[1,0].plot(epochs, res['train_losses'], label=f'{name} (Train)', alpha=0.7)
            axes[1,0].plot(epochs, res['val_losses'], label=f'{name} (Val)', alpha=0.9)
        axes[1,0].set_title('Training Curves - Top 3 Models'); axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('Loss'); axes[1,0].legend(); axes[1,0].set_yscale('log')

        for name, res in best_3:
            epochs = range(1, len(res['val_accuracies'])+1)
            axes[1,1].plot(epochs, res['train_accuracies'], label=f'{name} (Train)', alpha=0.7)
            axes[1,1].plot(epochs, res['val_accuracies'], label=f'{name} (Val)', alpha=0.9)
        axes[1,1].set_title('Accuracy Curves - Top 3 Models'); axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Accuracy'); axes[1,1].legend()
        axes[1,1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

        rankings = []
        for n in model_names:
            score = (1.0 / max(1e-8, results[n]['best_val_loss'])) * results[n]['val_accuracies'][-1]
            rankings.append((n, score))
        rankings.sort(key=lambda x: x[1], reverse=True)
        names, scores = zip(*rankings) if rankings else ([], [])

        axes[1,2].barh(range(len(names)), scores)
        axes[1,2].set_yticks(range(len(names))); axes[1,2].set_yticklabels(names)
        axes[1,2].set_title('Overall Model Ranking\n(Accuracy / Loss)'); axes[1,2].set_xlabel('Score')

        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/plots/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*80)
        for i, (name, score) in enumerate(rankings):
            res = results[name]
            logger.info(f"\n{i+1}. {name.upper()}")
            logger.info(f"   Best Val Loss: {res['best_val_loss']:.6f}")
            logger.info(f"   Final Val Acc: {res['val_accuracies'][-1]:.4f} ({res['val_accuracies'][-1]*100:.2f}%)")
            logger.info(f"   Parameters: {res['param_count']:,}")
            logger.info(f"   Score: {score:.4f}")

        if rankings:
            best_model = rankings[0][0]
            logger.info(f"\n🏆 BEST MODEL: {best_model.upper()}")
            logger.info(f"   Validation Loss: {results[best_model]['best_val_loss']:.6f}")
            logger.info(f"   Validation Accuracy: {results[best_model]['val_accuracies'][-1]:.4f}")
            logger.info(f"   Improvement over simple LSTM: {((results[best_model]['val_accuracies'][-1] - 0.5112) / 0.5112 * 100):+.2f}%")

    # ----------------------- Orchestration -------------------------
    def run_experiment(self):
        logger.info("Starting Advanced Training Experiment")
        logger.info("="*60)

        # Load Alpha data
        X, y, metadata = self.load_and_prepare_data_alpha()
        input_size = X.shape[2]  # Number of features
        
        # Extract metadata for batch processing
        vol = torch.ones_like(y) * 0.02  # Default volatility estimate
        ticker_ids = torch.zeros(len(y), dtype=torch.long)  # Placeholder for ticker IDs
        
        logger.info(f"Alpha dataset: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")

        results = self.compare_models(X, y, vol, input_size, ticker_ids)
        self.plot_comparison(results)
        try:
            mlflow.log_artifacts(f"{self.exp_dir}/plots", artifact_path="plots")
            comparison_json = f"{self.exp_dir}/model_comparison.json"
            if os.path.exists(comparison_json):
                mlflow.log_artifact(comparison_json)
        except Exception as e:
            logger.warning(f"MLflow artifact logging failed: {e}")

        final_time = datetime.now()
        total_h = (final_time - self.training_start_time).total_seconds() / 3600.0

        logger.info("="*80)
        logger.info("🎉 12-HOUR AI TRADING MODEL TRAINING COMPLETED!")
        logger.info("="*80)
        logger.info(f"🕒 Session Start: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🏁 Session End: {final_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  Total Training Time: {total_h:.2f} hours")
        logger.info(f"📁 Results Directory: {self.exp_dir}")
        logger.info(f"🤖 Models Trained: {len(results)}")

        if results:
            logger.info("\n📊 FINAL PERFORMANCE SUMMARY:")
            logger.info("="*50)
            sorted_results = sorted(results.items(),
                                    key=lambda x: x[1]['val_accuracies'][-1] if x[1]['val_accuracies'] else 0,
                                    reverse=True)
            for i, (name, res) in enumerate(sorted_results, 1):
                acc = res['val_accuracies'][-1] if res['val_accuracies'] else 0
                if acc > 0.60: status = "🟢 EXCELLENT"
                elif acc > 0.55: status = "🟡 VERY GOOD"
                elif acc > 0.52: status = "🔵 GOOD"
                else: status = "🔴 NEEDS WORK"
                logger.info(f"{i}. {name.upper()}: {acc:.4f} ({acc*100:.2f}%) {status}")

            best_name, best_res = sorted_results[0]
            best_acc = best_res['val_accuracies'][-1] if best_res['val_accuracies'] else 0
            logger.info("="*50)
            logger.info("🏆 CHAMPION MODEL:")
            logger.info(f"   🤖 {best_name.upper()}")
            logger.info(f"   🎯 Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
            logger.info(f"   📉 Best Loss: {best_res['best_val_loss']:.6f}")
            if best_acc > 0.60:
                logger.info("   🚀 READY FOR LIVE TRADING!")
            elif best_acc > 0.55:
                logger.info("   📊 STRONG PERFORMANCE - Ready for paper trading")
            else:
                logger.info("   🔧 Consider additional training or data")

        logger.info("="*80)
        logger.info("🎊 12-HOUR TRAINING SESSION COMPLETE!")
        logger.info("📈 Advanced AI trading models ready for deployment!")
        logger.info("="*80)

        if mlflow.active_run():
            mlflow.end_run()

        return results

    def train_single_model(self, model_name, config, max_epochs=2000, patience=50):
        """Train a single model with custom parameters"""
        logger.info(f"🏗️  Architecture: {config['type']}")
        logger.info(f"⚙️  Parameters: {config['params']}")

        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.log_param("model_type", config['type'])
            mlflow.log_params(config['params'])

            # Load Alpha data
            X, y, metadata = self.load_and_prepare_data_alpha()
            input_size = X.shape[2]

            # Extract metadata for batch processing
            vol = torch.ones_like(y) * 0.02  # Default volatility estimate
            ticker_ids = torch.zeros(len(y), dtype=torch.long)  # Placeholder for ticker IDs

            # Time-aware split
            X_train, X_val, y_train, y_val, vol_train, vol_val = self._time_split_by_ticker(
                X, y, vol, ticker_ids, val_frac=0.2
            )

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train, vol_train)
            val_dataset = TensorDataset(X_val, y_val, vol_val)
            batch_size = 8 if torch.cuda.is_available() else 16  # Reduced for stability
            pin_mem = torch.cuda.is_available()

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=pin_mem, persistent_workers=False
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=pin_mem, persistent_workers=False
            )

            # Create model
            model = create_advanced_model(config['type'], input_size, **config['params']).to(self.device)

            # Train with custom parameters
            result = self.train_model_custom(model, train_loader, val_loader, model_name, epochs=max_epochs, patience=patience)
            if result:
                mlflow.log_metric("best_val_loss", result['best_val_loss'])
                mlflow.log_metric("final_val_acc", result['final_val_acc'])
            return result

    def train_model_custom(self, model, train_loader, val_loader, model_name, epochs=2000, patience=50):
        """Modified train_model that accepts custom epochs and patience"""
        # This is a wrapper around the existing train_model method
        # Save original train_model logic but with custom parameters
        
        from models.advanced_models import FinancialLoss
        
        model = model.to(self.device)
        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_params:,}")
        
        # ANTI-OVERFITTING: Reduced learning rates and increased weight decay
        is_lstm = 'lstm' in model_name.lower()
        base_lr = 1e-4 if is_lstm else 5e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-3, betas=(0.9, 0.999))

        steps_per_epoch = max(1, len(train_loader))
        
        scheduler_cfg = dict(
            max_lr=base_lr * 10,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        logger.info("="*80)
        logger.info(f"🚀 STARTING TRAINING: {model_name.upper()}")
        logger.info("="*80)
        logger.info(f"💻 Device: {self.device} | AMP: {self.use_amp}")
        logger.info(f"🎯 Target Epochs: {epochs}")
        logger.info(f"📊 Model Parameters: {model_params:,}")
        logger.info(f"🔄 Batches/Epoch: {len(train_loader)} | Val Batches: {len(val_loader)}")
        
        # Use custom patience
        patience_counter = 0
        best_val_loss = float('inf')
        
        # Training metrics tracking
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        best_model_state = None
        
        # Resume from checkpoint if exists
        checkpoint_path = f"{self.exp_dir}/checkpoints/latest_{model_name}.pth"
        start_epoch = 0
        global_step = 0
        
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    # We'll recreate scheduler
                    pass
                if 'scaler_state_dict' in ckpt and self.use_amp:
                    self.scaler.load_state_dict(ckpt['scaler_state_dict'])
                
                start_epoch = int(ckpt.get('epoch', 0))
                global_step = int(ckpt.get('global_step', 0))
                best_val_loss = float(ckpt.get('best_val_loss', float('inf')))
                patience_counter = int(ckpt.get('patience_counter', 0))
                
                # Load metrics if available
                if 'train_losses' in ckpt:
                    train_losses = ckpt['train_losses']
                    val_losses = ckpt['val_losses'] 
                    train_accuracies = ckpt['train_accuracies']
                    val_accuracies = ckpt['val_accuracies']
                
                logger.info(f"Resuming {model_name} from checkpoint...")
                logger.info(f"✓ Resumed from epoch {start_epoch}, global_step {global_step}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                start_epoch, global_step = 0, 0

        last_epoch = global_step - 1 if global_step > 0 else -1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, last_epoch=last_epoch, **scheduler_cfg)
        
        # Training loop with custom epochs and patience
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_train_loss, epoch_train_acc = 0.0, 0.0
            
            # Training phase
            for batch_idx, (X_batch, y_batch, vol_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.autocast(device_type='cuda'):
                        outputs = model(X_batch)
                        # Handle different model output formats
                        if isinstance(outputs, dict):
                            pred = outputs['return_prediction']
                            vol_pred = outputs.get('volatility_prediction', None)
                            loss = FinancialLoss()(pred, y_batch, vol_pred)
                        else:
                            loss = FinancialLoss()(outputs, y_batch)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(X_batch)
                    # Handle different model output formats
                    if isinstance(outputs, dict):
                        pred = outputs['return_prediction']
                        vol_pred = outputs.get('volatility_prediction', None)
                        loss = FinancialLoss()(pred, y_batch, vol_pred)
                    else:
                        loss = FinancialLoss()(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                global_step += 1
                
                epoch_train_loss += float(loss.detach().cpu())
                
                # Calculate accuracy
                with torch.no_grad():
                    if isinstance(outputs, dict):
                        pred_values = outputs['return_prediction'].squeeze()
                    else:
                        pred_values = outputs.squeeze()
                    preds = torch.sigmoid(pred_values) > 0.5
                    targets_binary = (y_batch.squeeze() > 0).float()
                    acc = (preds == targets_binary).float().mean().item()
                    epoch_train_acc += acc
                
                # Save checkpoint every 100 steps
                if global_step % 100 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'patience_counter': patience_counter,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies,
                        'train_loss': loss.item()
                    }
                    if self.use_amp:
                        checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                    
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"  Latest checkpoint saved: {checkpoint_path}")
            
            # Validation phase
            model.eval()
            epoch_val_loss, epoch_val_acc = 0.0, 0.0
            
            with torch.no_grad():
                for X_batch, y_batch, vol_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    if self.use_amp:
                        with torch.autocast(device_type='cuda'):
                            outputs = model(X_batch)
                            # Handle different model output formats
                            if isinstance(outputs, dict):
                                pred = outputs['return_prediction']
                                vol_pred = outputs.get('volatility_prediction', None)
                                loss = FinancialLoss()(pred, y_batch, vol_pred)
                            else:
                                loss = FinancialLoss()(outputs, y_batch)
                    else:
                        outputs = model(X_batch)
                        # Handle different model output formats
                        if isinstance(outputs, dict):
                            pred = outputs['return_prediction']
                            vol_pred = outputs.get('volatility_prediction', None)
                            loss = FinancialLoss()(pred, y_batch, vol_pred)
                        else:
                            loss = FinancialLoss()(outputs, y_batch)
                    
                    epoch_val_loss += float(loss.detach().cpu())
                    
                    # Calculate accuracy
                    if isinstance(outputs, dict):
                        pred_values = outputs['return_prediction'].squeeze()
                    else:
                        pred_values = outputs.squeeze()
                    preds = torch.sigmoid(pred_values) > 0.5
                    targets_binary = (y_batch.squeeze() > 0).float()
                    acc = (preds == targets_binary).float().mean().item()
                    epoch_val_acc += acc
            
            # Average metrics
            epoch_train_loss /= len(train_loader)
            epoch_train_acc /= len(train_loader)
            epoch_val_loss /= len(val_loader)
            epoch_val_acc /= len(val_loader)
            
            # Store metrics
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accuracies.append(epoch_train_acc)
            val_accuracies.append(epoch_val_acc)

            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "train_acc": epoch_train_acc,
                "val_acc": epoch_val_acc
            }, step=epoch + 1)
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                elapsed_time = datetime.now() - self.training_start_time
                elapsed_hours = elapsed_time.total_seconds() / 3600
                remaining_hours = max(0, 12 - elapsed_hours)  # Assume 12-hour target
                
                logger.info("="*60)
                logger.info(f"📊 {model_name.upper()} - EPOCH {epoch+1}/{epochs}")
                logger.info("="*60)
                logger.info(f"⏰ Now: {datetime.now().strftime('%H:%M:%S')}")
                logger.info(f"⏱️  Elapsed: {elapsed_hours:.2f}h | Remaining: {remaining_hours:.2f}h")
                logger.info(f"🎯 Train Loss: {epoch_train_loss:.6f} → Val Loss: {epoch_val_loss:.6f}")
                logger.info(f"📈 Train Acc: {epoch_train_acc:.4f} → Val Acc: {epoch_val_acc:.4f}")
                logger.info(f"🔧 LR: {scheduler.get_last_lr()[0]:.2e} | Best Val Loss: {best_val_loss:.6f}")
                logger.info(f"⏳ Patience: {patience_counter}/{patience}")
            
            # Check for improvement
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                # Save best model
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies
                }
                torch.save(best_checkpoint, f"{self.exp_dir}/checkpoints/best_{model_name}.pth")
                logger.info(f"  ★ New best {model_name}! Val Loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping for {model_name} at epoch {epoch+1}")
                break
        
        # Return results
        if best_model_state is not None and train_losses and val_losses:
            try:
                best_ckpt = f"{self.exp_dir}/checkpoints/best_{model_name}.pth"
                if os.path.exists(best_ckpt):
                    mlflow.log_artifact(best_ckpt, artifact_path=model_name)
            except Exception as e:
                logger.warning(f"MLflow artifact logging failed: {e}")
            return {
                'best_val_loss': best_val_loss,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_train_acc': train_accuracies[-1],
                'final_val_acc': val_accuracies[-1],
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'param_count': model_params,
                'config': {
                    'type': 'custom',
                    'params': {'epochs': epochs, 'patience': patience}
                }
            }
        else:
            logger.error(f"Training failed for {model_name}")
            return None

    def run_specific_models(self, model_names, max_epochs=2000, patience=50, continue_remaining=True):
        """Train specific models, optionally continue with remaining models"""
        logger.info(f"🎯 Starting training for specific models: {model_names}")
        if continue_remaining:
            logger.info("🔄 Will automatically continue with remaining models after completion")
        
        results = {}
        
        # Train selected models first
        for model_name in model_names:
            if model_name not in self.model_configs:
                logger.warning(f"Unknown model: {model_name}. Skipping.")
                continue
            
            logger.info(f"="*80)
            logger.info(f"🤖 TRAINING SELECTED: {model_name.upper()}")
            logger.info(f"="*80)
            
            config = self.model_configs[model_name]
            result = self.train_single_model(model_name, config, max_epochs=max_epochs, patience=patience)
            
            if result:
                results[model_name] = result
                logger.info(f"✅ {model_name} completed successfully")
            else:
                logger.error(f"❌ {model_name} training failed")
        
        # Continue with remaining models if requested
        if continue_remaining:
            logger.info("="*80)
            logger.info("🔄 CONTINUING WITH REMAINING MODELS")
            logger.info("="*80)
            
            # Find models not yet trained
            completed_models = set(results.keys())
            if os.path.exists(f"{self.exp_dir}/model_comparison.json"):
                try:
                    with open(f"{self.exp_dir}/model_comparison.json", 'r') as f:
                        existing_results = json.load(f)
                    completed_models.update(existing_results.keys())
                except:
                    pass
            
            remaining_models = [name for name in self.model_configs.keys() 
                              if name not in completed_models]
            
            if remaining_models:
                logger.info(f"📋 Remaining models to train: {remaining_models}")
                for model_name in remaining_models:
                    logger.info(f"="*80)
                    logger.info(f"🤖 TRAINING REMAINING: {model_name.upper()}")
                    logger.info(f"="*80)
                    
                    config = self.model_configs[model_name]
                    result = self.train_single_model(model_name, config, max_epochs=max_epochs, patience=patience)
                    
                    if result:
                        results[model_name] = result
                        logger.info(f"✅ {model_name} completed successfully")
                    else:
                        logger.error(f"❌ {model_name} training failed")
            else:
                logger.info("🎉 All models already completed!")
        
        # Save results
        if results:
            results_file = f"{self.exp_dir}/model_comparison.json"
            
            # Load existing results if any
            existing_results = {}
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        existing_results = json.load(f)
                except:
                    pass
            
            # Merge with new results
            existing_results.update(results)
            
            with open(results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            logger.info(f"📊 Results saved to {results_file}")
            try:
                mlflow.log_artifact(results_file)
            except Exception as e:
                logger.warning(f"MLflow artifact logging failed: {e}")

        if mlflow.active_run():
            mlflow.end_run()

        return results

    # ----------------------- Check status --------------------------
    def check_models_status(self):
        logger.info("Checking models status...")

        if not self.checkpoint_data:
            logger.info("No saved models found")
            return None

        results_file = f"{self.exp_dir}/model_comparison.json"
        comparison = None
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    comparison = json.load(f)
                logger.info("Found model comparison results")
            except Exception as e:
                logger.warning(f"Error reading results: {e}")

        print("\n" + "="*80)
        print("📊 Models Status")
        print("="*80)

        model_info = []
        for name, ck in self.checkpoint_data.items():
            data = ck['data']; path = ck['path']
            epoch = data.get('epoch', 0)
            gstep = data.get('global_step', 0)
            train_loss = data.get('train_loss', 'N/A')
            detailed = comparison.get(name) if comparison and (name in comparison) else None
            model_info.append({
                'name': name,
                'epoch': epoch,
                'batch_counter': gstep,
                'train_loss': train_loss,
                'checkpoint_path': os.path.basename(path),
                'detailed_results': detailed
            })

        if comparison:
            model_info.sort(key=lambda x: x['detailed_results']['best_val_loss'] if x['detailed_results'] else float('inf'))

        for i, info in enumerate(model_info):
            print(f"\n{i+1}. 🤖 {info['name'].upper()}")
            print(f"   📁 File: {info['checkpoint_path']}")
            print(f"   🔄 Epoch: {info['epoch']}")
            print(f"   📊 Batches: {info['batch_counter']:,}")
            if info['detailed_results']:
                res = info['detailed_results']
                print(f"   🎯 Validation Accuracy: {res['final_val_acc']:.4f} ({res['final_val_acc']*100:.2f}%)")
                print(f"   📉 Best Val Loss: {res['best_val_loss']:.6f}")
                print(f"   ⚙️  Parameters: {res['param_count']:,}")
                acc = res['final_val_acc']
                quality = "🟢 Excellent" if acc > 0.55 else ("🟡 Good" if acc > 0.52 else "🔴 Needs Improvement")
                print(f"   📈 Quality: {quality}")
            else:
                print(f"   ⚠️  Last Loss: {info['train_loss']}")
                print(f"   📝 Status: Training in progress")

        print(f"\n{'='*80}")
        print("💡 Recommendations:")
        if not comparison:
            print("• Training not yet completed - can continue from last checkpoint")
            print("• Run script again to continue training")
        else:
            best = model_info[0] if model_info else None
            if best and best['detailed_results']:
                best_acc = best['detailed_results']['final_val_acc']
                print(f"• Best model: {best['name']}")
                print(f"• Accuracy: {best_acc*100:.2f}%")
                if best_acc > 0.55:
                    print("• Model ready for production! 🚀")
                elif best_acc > 0.52:
                    print("• Model is decent - consider additional training or parameter tuning")
                else:
                    print("• Model needs improvement - try more data or different architecture")

        print("="*80)
        return model_info

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Advanced Financial Model Training (Upgraded)')
    parser.add_argument('--experiment', type=str, default="financial_models_comparison",
                        help='Experiment name')
    parser.add_argument('--check', action='store_true', help='Check existing models status only')
    parser.add_argument('--no_causal_mask', action='store_true', help='Disable causal masking for Transformer')
    parser.add_argument('--model', action='append', help='Specific model(s) to train (can be used multiple times)')
    parser.add_argument('--max-epochs', type=int, default=2000, help='Maximum epochs per model')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--no-continue', action='store_true', help='Do not automatically continue with remaining models')
    parser.add_argument('--label-mode', type=str, default='alpha', choices=['alpha', 'cls', 'rank'],
                        help='Label mode: alpha (regression), cls (classification), rank (ranking)')
    parser.add_argument('--prediction-horizon', type=int, default=1, choices=[1, 5, 20],
                        help='Prediction horizon in days')
    parser.add_argument('--neutral-zone-bps', type=float, default=5.0,
                        help='Neutral zone in basis points for classification')
    parser.add_argument('--conformal', action='store_true',
                        help='Enable conformal prediction for uncertainty quantification')
    parser.add_argument('--risk-aware-loss', type=str, choices=['sharpe', 'sortino', 'calmar', 'composite'],
                        help='Use risk-aware loss function for training')
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['lstm', 'trans', 'patchtst', 'itransformer', 'tsmixer', 'ensemble', 'wavenet'],
                        help='Specific model architectures to train')
    args = parser.parse_args()

    trainer = AdvancedTrainer(
        args.experiment, 
        use_causal_mask=not args.no_causal_mask,
        label_mode=args.label_mode,
        prediction_horizon=args.prediction_horizon,
        neutral_zone_bps=args.neutral_zone_bps
    )

    if args.check:
        result = trainer.check_models_status()
        if mlflow.active_run():
            mlflow.end_run()
        return result

    # If specific models requested, train only those
    if args.model:
        logger.info(f"Training specific models: {args.model}")
        continue_remaining = not args.no_continue
        results = trainer.run_specific_models(args.model, max_epochs=args.max_epochs, patience=args.patience, continue_remaining=continue_remaining)
        return results

    logger.info("Testing advanced model architectures...")
    # Optional quick self-test of architectures (keep your original behavior):
    os.system("python advanced_models.py")

    results = trainer.run_experiment()
    return results

if __name__ == "__main__":
    main()
