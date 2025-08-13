import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def build_edge_index_from_returns(returns_df, threshold=0.4):
    corr = returns_df.corr().fillna(0)
    n = corr.shape[0]
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr.iloc[i,j]) >= threshold:
                edge_list.append([i,j]); edge_list.append([j,i])
    if not edge_list:
        for i in range(n):
            for j in range(i+1, n):
                edge_list.append([i,j]); edge_list.append([j,i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

class UniverseDataset(torch.utils.data.Dataset):
    def __init__(self, all_data_dict, tickers_list, seq_len=30, feature_cols=None, top_n=30, corr_threshold=0.4):
        self.all_data = all_data_dict
        self.tickers = tickers_list[:top_n]
        self.seq_len = seq_len
        self.feature_cols = feature_cols or ["Open","High","Low","Close","Volume","sentiment_score","SMA_10","SMA_50","RSI_14","MACD"]
        self.corr_threshold = corr_threshold
        dates = list(self.all_data[self.tickers[0]].index)
        self.index_dates = dates[self.seq_len:-2]
    def __len__(self):
        return len(self.index_dates)
    def __getitem__(self, idx):
        date = self.index_dates[idx]
        seqs = []
        returns = []
        for t in self.tickers:
            df = self.all_data[t]
            pos = df.index.get_loc(date)
            window = df.iloc[pos-self.seq_len:pos]
            seq = window[self.feature_cols].values.astype(np.float32)
            seqs.append(seq)
            try:
                nxt = df.iloc[pos+1]["Close"]; cur = df.iloc[pos]["Close"]
                r = (nxt - cur) / (cur + 1e-8)
            except Exception:
                r = 0.0
            returns.append(r)
        seqs = np.stack(seqs)
        daily_slice = {t: self.all_data[t].iloc[pos-self.seq_len:pos] for t in self.tickers}
        returns_df = pd.DataFrame({t: d["Close"].pct_change().fillna(0).values for t,d in daily_slice.items()})
        edge_index = build_edge_index_from_returns(returns_df, threshold=self.corr_threshold)
        x = torch.tensor(seqs, dtype=torch.float)
        y = torch.tensor(returns, dtype=torch.float)
        data = Data()
        data.seq = x
        data.y = y
        data.edge_index = edge_index
        data.num_nodes = len(self.tickers)
        return data
