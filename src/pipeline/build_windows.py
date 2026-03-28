import numpy as np
import pandas as pd
from tqdm import tqdm

from config import WINDOW, ACTOR2_START

def build_windows():
    df        = pd.read_csv("savedWork/cleaned_data.csv", parse_dates=["Timestamp"])
    ts_raw    = df["Timestamp"].values
    lbl_raw   = df["label"].values
    sorted_cols = [c for c in df.columns if c not in ("Timestamp", "label")]
    values    = df[sorted_cols].values.astype(np.float32)
    N_total   = len(values)

    print(f"Building windows  N={N_total:,}  W={WINDOW} ...")

    X_all, y_all, ts_all, lbl_all = [], [], [], []

    for i in tqdm(range(N_total - WINDOW)):
        w = values[i : i + WINDOW]
        X_all.append(w[:-1])
        y_all.append(w[-1])
        ts_all.append(ts_raw[i + WINDOW - 1])
        lbl_all.append(lbl_raw[i + WINDOW - 1])

    X_all   = np.array(X_all,  dtype=np.float32)
    y_all   = np.array(y_all,  dtype=np.float32)
    ts_all  = np.array(ts_all)
    lbl_all = np.array(lbl_all, dtype=np.int64)

    ts_pd       = pd.to_datetime(ts_all)
    train_mask  = ts_pd < pd.Timestamp(ACTOR2_START)

    X_train  = X_all[train_mask]
    y_train  = y_all[train_mask]
    ts_train = ts_all[train_mask]

    print(f"  TRAIN (Actor 1): {len(X_train):,} windows — 0 anomalies")
    print(f"  EVAL  (all)    : {len(X_all):,} windows  "
        f"— {lbl_all.sum():,} anomalies ({lbl_all.mean()*100:.1f}%)")

    np.save("savedWork/X_train.npy",     X_train)
    np.save("savedWork/y_train.npy",     y_train)
    np.save("savedWork/X_eval.npy",      X_all)
    np.save("savedWork/y_eval.npy",      y_all)
    np.save("savedWork/ts_eval.npy",     ts_all)
    np.save("savedWork/labels_eval.npy", lbl_all)
    print("  Arrays saved.")

    INPUT_DIM  = WINDOW - 1    # 29

if __name__ == "__main__":
    build_windows()
