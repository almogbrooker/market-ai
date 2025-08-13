import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import UniverseDataset
from model import HybridTransformerGAT
from config import DEVICE

def load_checkpoint(path, num_features):
    ck = torch.load(path, map_location=DEVICE)
    model = HybridTransformerGAT(num_features=num_features, d_model=64, nhead=4, dim_feedforward=128, nlayers=6, gat_hidden=64, gat_out=32, dropout=0.3, device=DEVICE)
    model.load_state_dict(ck["model_state"]); model.to(DEVICE); model.eval()
    return model

def predict_on_dataset(model, ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    preds=[]; trues=[]
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch).cpu().numpy()
        preds.append(out); trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    return preds, trues
