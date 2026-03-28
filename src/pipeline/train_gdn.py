import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from config import nbr_sensors, in_dim, GDN_BATCH, GDN_EPOCHS, GDN_LR, GDN_PATIENCE
from src.models.model_gdn import GDN

def train_gdn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    train_ds     = torch.load("saved_ds/gdn_train.pt", weights_only=False)
    train_loader = DataLoader(train_ds, batch_size=GDN_BATCH, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    gdn_model = GDN(num_sensors=nbr_sensors, input_dim=in_dim).to(device)
    gdn_opt   = torch.optim.Adam(gdn_model.parameters(), lr=GDN_LR, weight_decay=1e-4)
    gdn_sched = torch.optim.lr_scheduler.CosineAnnealingLR(gdn_opt, T_max=GDN_EPOCHS)
    criterion = nn.MSELoss()

    print(f"  GDN parameters  : {sum(p.numel() for p in gdn_model.parameters()):,}")

    gdn_history  = []
    best_loss, pat_ctr = float("inf"), 0

    pbar = tqdm(range(1, GDN_EPOCHS + 1), desc="GDN Training", unit="epoch")
    for epoch in pbar:
        gdn_model.train()
        total = 0.0
        for batch in train_loader:
            batch  = batch.to(device, non_blocking=True)
            pred   = gdn_model(batch)
            loss   = criterion(pred, batch.y)
            gdn_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(gdn_model.parameters(), 1.0)
            gdn_opt.step()
            total += loss.item()
        gdn_sched.step()
        avg = total / len(train_loader)
        gdn_history.append(avg)
        pbar.set_postfix({"MSE": f"{avg:.5f}"})
        if avg < best_loss:
            best_loss, pat_ctr = avg, 0
            torch.save(gdn_model.state_dict(), "savedWork/gdn_best.pt")
        else:
            pat_ctr += 1
            if pat_ctr >= GDN_PATIENCE:
                tqdm.write(f"GDN early stop epoch {epoch}. Best: {best_loss:.5f}")
                break

    print(f"\n  GDN best MSE : {best_loss:.5f}")

    # Collect per-sensor error statistics from Actor 1
    print("Computing GDN training error statistics...")
    gdn_model.load_state_dict(torch.load("savedWork/gdn_best.pt", weights_only=True))
    gdn_model.eval()
    all_err = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(train_ds, batch_size=512, shuffle=False),
                          desc="GDN stats"):
            batch = batch.to(device)
            B     = int(batch.batch.max().item()) + 1
            pred  = gdn_model(batch).view(B, 94)
            err   = (pred - batch.y.view(B, 94)).abs()
            all_err.append(err.cpu())
    all_err  = torch.cat(all_err, dim=0)
    gdn_mean = all_err.mean(dim=0)
    gdn_std  = all_err.std(dim=0)
    torch.save({"mean": gdn_mean, "std": gdn_std}, "savedWork/gdn_err_stats.pt")
    print(f"  Mean MAE: {gdn_mean.mean():.5f}  Std: {gdn_std.mean():.5f}")


if __name__ == "__main__":
    train_gdn()
