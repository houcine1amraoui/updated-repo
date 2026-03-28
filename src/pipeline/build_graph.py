import numpy as np

from config import correlation_threshold

def build_graph():
    X_train = np.load("savedWork/X_train.npy")
    N, W, S = X_train.shape

    flat = X_train.reshape(-1, S)
    corr = np.corrcoef(flat.T)                    # (94, 94)

    CORR_THRESHOLD = 0.5
    corr = np.nan_to_num(corr, nan=0.0)

    # Build edge list directly — no dense adjacency matrix stored
    rows, cols = np.where(np.abs(corr) > CORR_THRESHOLD)

    # Remove self-loops
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]

    # edge_list shape: (2, E)  — row 0 = sources, row 1 = destinations
    edge_list = np.stack([rows, cols], axis=0).astype(np.int64)

    n_edges = edge_list.shape[1] // 2   # undirected count
    density = n_edges / (S * (S - 1) / 2) * 100

    print(f"  Sensors   : {S}")
    print(f"  Edges     : {n_edges}  (threshold={CORR_THRESHOLD})")
    print(f"  Density   : {density:.1f}%")
    print(f"  edge_list : {edge_list.shape}  (2 × {edge_list.shape[1]} directed edges)")

    np.save("savedWork/edge_list.npy", edge_list)
    print("  Saved → savedWork/edge_list.npy")

if __name__ == "__main__":
    build_graph()
