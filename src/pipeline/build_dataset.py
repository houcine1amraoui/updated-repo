import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from config import nbr_sensors, correlation_threshold

X_train  = np.load("savedWork/X_train.npy")
y_train  = np.load("savedWork/y_train.npy")
X_eval   = np.load("savedWork/X_eval.npy")
y_eval   = np.load("savedWork/y_eval.npy")
ts_eval  = np.load("savedWork/ts_eval.npy",     allow_pickle=True)
lbl_eval = np.load("savedWork/labels_eval.npy")

NUM_SENSORS    = nbr_sensors
CORR_THRESHOLD = correlation_threshold


def window_edge_index(window, threshold=CORR_THRESHOLD):
    """Per-window dynamic edge list from Pearson correlation of that 29-step slice."""
    corr = np.corrcoef(window.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Extract edges directly — no intermediate adjacency matrix
    rows, cols = np.where(np.abs(corr) > threshold)

    # Remove self-loops
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]

    if len(rows) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)


def make_dynamic_dataset(X_arr, y_arr, ts_arr=None, lbl_arr=None, desc=""):
    dataset  = []
    has_meta = ts_arr is not None
    for idx in tqdm(range(len(X_arr)), desc=desc):
        xi = X_arr[idx]
        yi = y_arr[idx]
        ei = window_edge_index(xi)
        d  = Data(
            x          = torch.tensor(xi.T, dtype=torch.float),
            y          = torch.tensor(yi,   dtype=torch.float),
            edge_index = ei
        )
        if has_meta:
            d.ts    = str(ts_arr[idx])
            d.label = int(lbl_arr[idx])
        dataset.append(d)
    return dataset


print("Building train dataset (dynamic graphs)...")
train_ds = make_dynamic_dataset(X_train, y_train, desc="Train")
torch.save(train_ds, "saved_ds/gdn_train.pt")

print("Building eval dataset (dynamic graphs)...")
eval_ds = make_dynamic_dataset(X_eval, y_eval, ts_eval, lbl_eval, desc="Eval ")
torch.save(eval_ds, "saved_ds/gdn_eval.pt")

print(f"\n  Train : {len(train_ds):,} graphs")
print(f"  Eval  : {len(eval_ds):,} graphs")
print(f"  Sample x shape : {train_ds[0].x.shape}  (94 × 29)")
