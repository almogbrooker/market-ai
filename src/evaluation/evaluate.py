import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import UniverseDataset
from model import HybridTransformerGAT
from utils.config import trading_settings as settings


def load_checkpoint(path, num_features):
    ck = torch.load(path, map_location=settings.device)
    model = HybridTransformerGAT(
        num_features=num_features,
        d_model=settings.d_model,
        nhead=settings.nhead,
        dim_feedforward=128,
        nlayers=settings.transformer_layers,
        gat_hidden=settings.gat_hidden,
        gat_out=settings.gat_out,
        dropout=settings.dropout,
        device=settings.device,
    )
    model.load_state_dict(ck["model_state"])
    model.to(settings.device)
    model.eval()
    return model


def predict_on_dataset(model, ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    preds = []
    trues = []
    for batch in loader:
        batch = batch.to(settings.device)
        with torch.no_grad():
            out = model(batch).cpu().numpy()
        preds.append(out)
        trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues

