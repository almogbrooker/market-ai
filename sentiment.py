
import re
from typing import Optional, Tuple
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "yiyanghkust/finbert-tone"

_tokenizer = None
_model = None

def _load() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model

def _clean(t: str) -> str:
    if not isinstance(t, str): return ""
    t = re.sub(r'http\S+|www\.\S+', '', t)
    t = re.sub(r'[^\w\s\.,!\?-]', ' ', t)
    return ' '.join(t.split())

def analyze_news_sentiment(news_df: pd.DataFrame, title_col: str = "title",
                           desc_col: str = "description", batch_size:int=32,
                           device: Optional[str] = None) -> pd.DataFrame:
    """Adds sentiment columns using FinBERT. Logit order is [neg, neu, pos]."""
    if news_df is None or news_df.empty:
        return news_df

    tok, mdl = _load()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)

    titles = news_df.get(title_col, pd.Series(index=news_df.index, dtype=object)).fillna("")
    descs  = news_df.get(desc_col,  pd.Series(index=news_df.index, dtype=object)).fillna("")
    texts = (titles.astype(str) + ". " + descs.astype(str)).apply(_clean).tolist()

    negs, neus, poss = [], [], []
    labels = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1)  # [B, 3] -> [neg, neu, pos]
        p = probs.detach().cpu().numpy()
        negs.extend(p[:,0].tolist())
        neus.extend(p[:,1].tolist())
        poss.extend(p[:,2].tolist())
        idx = probs.argmax(dim=-1).detach().cpu().tolist()
        labels.extend(["negative" if a==0 else "neutral" if a==1 else "positive" for a in idx])

    out = news_df.copy()
    out["sentiment_neg"] = negs
    out["sentiment_neu"] = neus
    out["sentiment_pos"] = poss
    out["sentiment_label"] = labels
    out["sentiment_compound"] = out["sentiment_pos"] - out["sentiment_neg"]
    return out

class SentimentAnalyzer:
    """Thin wrapper to match existing imports in main_enhanced.py"""
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def analyze(self, news_df: pd.DataFrame, title_col: str = "title", desc_col: str = "description") -> pd.DataFrame:
        return analyze_news_sentiment(news_df, title_col=title_col, desc_col=desc_col, device=self.device)
