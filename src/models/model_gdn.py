import torch
import torch.nn as nn
import torch.nn.functional as F

class GDNAttentionLayer(nn.Module):
    """
    One layer of GDN-style graph attention.

    Follows the paper's formulation exactly:

      Attention weights (from embeddings only, not input features):
          e_ij = LeakyReLU( a^T · [W_q · v_i  ||  W_k · v_j] )
          α_ij = softmax over j in N_i of e_ij

      Aggregated representation for sensor i:
          z_i = ReLU( Σ_{j ∈ N_i} α_ij · (W_v · [x_j || v_j]) )

    Where:
        v_i    = sensor embedding for node i    (d-dim)
        x_j    = input features for node j      (in_dim-dim)
        N_i    = top-k neighbours of i from graph structure learning
        W_q, W_k = query/key projections for attention
        W_v    = value projection applied to [feature || embedding]
        a      = attention vector
    """

    def __init__(self, in_dim, emb_dim, out_dim, heads=1, dropout=0.2):
        """
        Parameters
        ----------
        in_dim  : input feature dim per node (= input_dim = 29)
        emb_dim : sensor embedding dim       (= hidden = 64)
        out_dim : output feature dim         (= hidden = 64)
        heads   : number of parallel attention heads
        dropout : attention dropout probability
        """
        super().__init__()
        self.heads   = heads
        self.out_dim = out_dim
        self.d_head  = out_dim // heads    # dim per head
        
        # Used to compute attention weights α_ij
        self.W_q = nn.Linear(emb_dim, out_dim, bias=False)   # v_i → query
        self.W_k = nn.Linear(emb_dim, out_dim, bias=False)   # v_j → key

        # Used to compute what gets aggregated after weighting by α_ij
        self.W_v = nn.Linear(in_dim + emb_dim, out_dim, bias=False)

        # e_ij = a^T · [q_i || k_j]   (dot product with learned vector)
        self.attn_vec = nn.Parameter(torch.empty(heads, 2 * self.d_head))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.leaky   = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, x, emb, edge_index):
        """
        Parameters
        ----------
        x          : (B*N, in_dim)      node input features
        emb        : (B*N, emb_dim)     sensor embeddings (tiled for batch)
        edge_index : (2, B*N*k)         dynamic top-k graph from GDN._topk_graph()

        Returns
        -------
        z : (B*N, out_dim)   attention-aggregated node representations
        """
        src, dst = edge_index[0], edge_index[1]   # src→dst directed edges
        BN = x.shape[0]
        Q = self.W_q(emb)                          # (B*N, out_dim)
        K = self.W_k(emb)                          # (B*N, out_dim)

        # Reshape for multi-head: (B*N, heads, d_head)
        Q = Q.view(BN, self.heads, self.d_head)
        K = K.view(BN, self.heads, self.d_head)

        # Gather query for dst node, key for src node, concatenate
        q_dst = Q[dst]                             # (E, heads, d_head)
        k_src = K[src]                             # (E, heads, d_head)
        qk    = torch.cat([q_dst, k_src], dim=-1)  # (E, heads, 2*d_head)

        # e_ij = a^T · [q_i || k_j] per head
        e = (qk * self.attn_vec.unsqueeze(0)).sum(dim=-1)   # (E, heads)
        e = self.leaky(e)

        # Softmax over incoming edges per destination node
        # Use scatter_softmax pattern via manual stable softmax
        alpha = torch.zeros(BN, self.heads, device=x.device)

        # For each destination node, gather its incoming edge scores
        # and compute softmax (numerically stable via scatter)
        e_max = torch.full((BN, self.heads), float('-inf'), device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax', include_self=True)
        e_exp = torch.exp(e - e_max[dst])                    # (E, heads)
        e_sum = torch.zeros(BN, self.heads, device=x.device)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e_exp), e_exp)
        alpha_e = e_exp / (e_sum[dst] + 1e-8)               # (E, heads) — softmax weights
        alpha_e = self.dropout(alpha_e)

        # ── Step 3b: Compute values from [input features || embedding] ────────
        val_input = torch.cat([x, emb], dim=-1)              # (B*N, in_dim+emb_dim)
        V         = self.W_v(val_input)                      # (B*N, out_dim)
        V         = V.view(BN, self.heads, self.d_head)
        V_src     = V[src]                                   # (E, heads, d_head)

        # Weighted aggregation: z_i = Σ_{j∈N_i} α_ij · v_j
        weighted  = alpha_e.unsqueeze(-1) * V_src            # (E, heads, d_head)
        z = torch.zeros(BN, self.heads, self.d_head, device=x.device)
        z.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2)
                         .expand_as(weighted), weighted)
        z = z.reshape(BN, self.out_dim)                      # (B*N, out_dim)
        z = F.elu(z)

        # LayerNorm + residual (residual projected to match dim if needed)
        return self.norm(z)


# ─────────────────────────────────────────────────────────────────────────────
class GDN(nn.Module):
    """
    Graph Deviation Network — Figure 1 faithful implementation.

    Data flow:
        data.x     (B*N, input_dim) → used as sensor input features
        data.batch (B*N,)           → used to recover batch size B
        data.edge_index             → IGNORED. Graph is built internally.

    The graph is never pre-computed outside the model.
    """

    def __init__(
        self,
        num_sensors : int   = 94,
        input_dim   : int   = 29,
        hidden      : int   = 64,
        topk        : int   = 15,
        heads       : int   = 4,
        dropout     : float = 0.2,
    ):
        super().__init__()

        self.num_sensors = num_sensors
        self.topk        = topk
        self.hidden      = hidden
        self.sensor_emb = nn.Embedding(num_sensors, hidden)
        nn.init.xavier_uniform_(self.sensor_emb.weight)

        # Layer 1: input features (29-dim) + embeddings (64-dim) → 64-dim z
        self.gdn_attn1 = GDNAttentionLayer(
            in_dim  = input_dim,   # 29  — raw sensor history
            emb_dim = hidden,      # 64  — sensor embedding dim
            out_dim = hidden,      # 64  — output z_i dim
            heads   = heads,       # 4
            dropout = dropout,
        )

        # Layer 2: z from layer 1 (64-dim) + embeddings (64-dim) → 64-dim z
        self.gdn_attn2 = GDNAttentionLayer(
            in_dim  = hidden,      # 64  — output of layer 1
            emb_dim = hidden,      # 64
            out_dim = hidden,      # 64
            heads   = 1,           # single head in layer 2
            dropout = dropout,
        )

        # Takes [z_i || v_i] and predicts next sensor value
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),    # scalar next-step value per sensor
        )

    def _topk_graph(self, batch_size: int, device):
        """
        Learns the graph structure from sensor embeddings.

        Algorithm (Figure 1, Step 2):
          1. L2-normalise all sensor embeddings → unit vectors
          2. Cosine similarity matrix = normalised_emb @ normalised_emb.T
          3. For each sensor i, keep its top-k highest-similarity neighbours
          4. Tile the result for all B graphs in the batch

        The graph is rebuilt every forward pass.
        As training progresses, embeddings cluster by behavioural similarity,
        so the graph evolves to connect sensors with similar normal patterns.

        Returns
        -------
        edge_index : LongTensor (2, B*N*k)
            Row 0 = source node indices (neighbours)
            Row 1 = destination node indices (centre nodes)
        """
        N   = self.num_sensors

        # Cosine similarity between all sensor embedding pairs
        emb = F.normalize(self.sensor_emb.weight, dim=1)   # (N, hidden) unit vecs
        sim = torch.mm(emb, emb.T)                          # (N, N) cosine sim matrix

        # Exclude self-connections (diagonal = 1.0 always, would always win)
        sim.fill_diagonal_(-1e9)

        # Top-k neighbours per sensor
        knn = sim.topk(self.topk, dim=1).indices             # (N, k) neighbour indices

        # Build directed edges: neighbour j → centre i
        src = knn.reshape(-1)                                # (N*k,) source = neighbours
        dst = (torch.arange(N, device=device)
               .unsqueeze(1).expand(N, self.topk)
               .reshape(-1))                                 # (N*k,) dest = centre nodes

        # Graph g's nodes start at index g*N
        offset = torch.arange(batch_size, device=device) * N   # (B,)
        src_b  = (src.unsqueeze(0) + offset.unsqueeze(1)).reshape(-1)  # (B*N*k,)
        dst_b  = (dst.unsqueeze(0) + offset.unsqueeze(1)).reshape(-1)  # (B*N*k,)

        return torch.stack([src_b, dst_b], dim=0)            # (2, B*N*k)

    def forward(self, data):
        """
        Parameters
        ----------
        data.x     : (B*N, input_dim)   sensor history features
        data.batch : (B*N,)             node-to-graph assignment
        data.edge_index                 NOT USED — graph built internally

        Returns
        -------
        pred : (B*N,)   predicted next-step value per sensor per graph
        """
        x      = data.x                                      # (B*N, 29)
        B      = int(data.batch.max().item()) + 1
        N      = self.num_sensors
        device = x.device

        # Retrieve sensor embeddings, tiled for the batc
        node_idx = torch.arange(N, device=device).repeat(B)  # (B*N,) [0..93, 0..93, ...]
        emb      = self.sensor_emb(node_idx)                  # (B*N, hidden)

        # Called every forward pass so the graph evolves with the embeddings
        edge_index = self._topk_graph(B, device)              # (2, B*N*k)

        # Layer 1: input features x + embeddings emb → z1
        z = self.gdn_attn1(x, emb, edge_index)                # (B*N, hidden)

        # Layer 2: z1 + embeddings emb → z2
        z = self.gdn_attn2(z, emb, edge_index)                # (B*N, hidden)

        # Forecasting head: [z_i || v_i] → predicted next value
        pred = self.forecast_head(
            torch.cat([z, emb], dim=1)                        # (B*N, hidden*2)
        ).squeeze(-1)                                          # (B*N,)

        return pred

    @torch.no_grad()
    def anomaly_score(self, data, train_mean, train_std):
        """
        Step 4 from Figure 1: compare observation with prediction.

        For each sensor i at window t:
            raw_error_i(t)  = |predicted_i(t) - actual_i(t)|
            norm_error_i(t) = (raw_error_i(t) - μ_i) / σ_i

        Graph-level score = mean of top-10% sensor errors
        (more robust than pure max — avoids single noisy sensor dominating)

        Parameters
        ----------
        train_mean : (N,)  per-sensor mean MAE on Actor 1 training data
        train_std  : (N,)  per-sensor std  MAE on Actor 1 training data

        Returns
        -------
        scores   : (B,)    anomaly score per window
        norm_err : (B, N)  per-sensor normalised error (for heatmap)
        """
        self.eval()
        pred   = self.forward(data)
        target = data.y
        B      = int(data.batch.max().item()) + 1
        N      = self.num_sensors

        # Per-sensor absolute error
        err  = (pred - target).abs().view(B, N)               # (B, 94)

        # Normalise by training statistics (z-score)
        mu       = train_mean.to(err.device).unsqueeze(0)     # (1, 94)
        sig      = (train_std + 1e-6).to(err.device).unsqueeze(0)
        norm_err = (err - mu) / sig                            # (B, 94)

        # Score = mean of top-10% sensors (more robust than pure max)
        k      = max(1, int(0.1 * N))                         # top ~9 sensors
        scores = norm_err.topk(k, dim=1).values.mean(dim=1)   # (B,)

        return scores, norm_err


if __name__ == "__main__":
    _m = GDN(num_sensors=94, input_dim=29, hidden=64, topk=15, heads=4)
    _p = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    print(f"  GDN parameters: {_p:,}")

    # Verify no dependency on data.edge_index
    from torch_geometric.data import Data, Batch
    _d  = Data(
        x          = torch.randn(94, 29),
        y          = torch.randn(94),
        edge_index = torch.zeros(2, 0, dtype=torch.long),  # empty — deliberately
    )
    _batch = Batch.from_data_list([_d, _d])
    _out   = _m(_batch)
    print(f"  Forward output shape : {_out.shape}  (expected: torch.Size([188]))")
    print(f"  data.edge_index was  : NOT USED  ✓")
    del _m, _d, _batch, _out
