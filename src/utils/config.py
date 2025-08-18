# config.py
import os

# Data / API
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
TICKERS = ["AAPL","MSFT","GOOG","AMZN","NVDA","TSLA","META","AMD","INTC","QCOM"]
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "YOUR_NEWSAPI_KEY_HERE")

# Universe & windows
TOP_N = 30
TIME_WINDOW = 30
SEQ_LEN = 30

# Training / runtime
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
BATCH_SIZE = 8
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
PATIENCE = 8
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
DATA_DIR = "data"
LOG_DIR = "runs"

# Model hyperparams
D_MODEL = 64
NHEAD = 4
TRANSFORMER_LAYERS = 6
GAT_HIDDEN = 64
GAT_OUT = 32
DROPOUT = 0.3

# ensure dirs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
