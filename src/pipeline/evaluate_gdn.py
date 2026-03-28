import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import nbr_sensors, in_dim, ACTOR2_START, ACTOR2_END
from src.models.model_gdn import GDN

ACTOR1_END    = "2022-11-07"
ACTOR1_RETURN = "2022-11-11"

def evaluate_gdn():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gdn_model = GDN(num_sensors=nbr_sensors, input_dim=in_dim).to(device)
        gdn_model.load_state_dict(torch.load("savedWork/gdn_best.pt", weights_only=True))
        gdn_model.eval()

        stats    = torch.load("savedWork/gdn_err_stats.pt", weights_only=True)
        gdn_mean = stats["mean"]
        gdn_std  = stats["std"]

        eval_ds     = torch.load("saved_ds/gdn_eval.pt", weights_only=False)
        eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False, num_workers=0)

        gdn_scores     = []
        gdn_sensor_err = []

        with torch.no_grad():
                for batch in tqdm(eval_loader, desc="GDN Eval"):
                        batch = batch.to(device)
                        sc, nerr = gdn_model.anomaly_score(batch, gdn_mean, gdn_std)
                        gdn_scores.append(sc.cpu())
                        gdn_sensor_err.append(nerr.cpu())

        gdn_scores     = torch.cat(gdn_scores).numpy()
        gdn_sensor_err = torch.cat(gdn_sensor_err).numpy()

        all_labels = np.array([d.label for d in eval_ds])
        all_ts     = pd.to_datetime([d.ts     for d in eval_ds])

        # Optimal threshold via Youden's J
        fpr, tpr, thresholds = roc_curve(all_labels, gdn_scores)
        gdn_threshold = float(thresholds[np.argmax(tpr - fpr)])
        gdn_preds = (gdn_scores > gdn_threshold).astype(int)

        gdn_auroc = roc_auc_score(all_labels, gdn_scores)
        gdn_auprc = average_precision_score(all_labels, gdn_scores)
        gdn_fpr   = (gdn_preds[all_labels == 0] == 1).mean()

        print("═" * 55)
        print(f"  GDN  AUROC : {gdn_auroc:.4f}")
        print(f"  GDN  AUPRC : {gdn_auprc:.4f}")
        print(f"  GDN  FPR   : {gdn_fpr:.4f}  (threshold={gdn_threshold:.4f})")
        print("═" * 55)
        print(classification_report(all_labels, gdn_preds,
        target_names=["Normal (Actor 1)", "Anomaly (Actor 2)"]))

        # Confusion matrix
        cm = confusion_matrix(all_labels, gdn_preds)
        ConfusionMatrixDisplay(cm, display_labels=["Actor 1", "Actor 2"]).plot(cmap="Blues")
        plt.title(f"GDN Confusion Matrix  |  AUROC={gdn_auroc:.4f}")
        plt.tight_layout()
        plt.savefig("savedWork/gdn_confusion_matrix.png", dpi=150)
        plt.show()

        # Timeline plot
        A2_START = pd.Timestamp(ACTOR2_START)
        A2_END   = pd.Timestamp(ACTOR2_END)

        fig, axes = plt.subplots(2, 1, figsize=(16, 7))
        ax = axes[0]
        ax.scatter(all_ts[all_labels==0], gdn_scores[all_labels==0],
                s=1, c="steelblue", alpha=0.3, label="Normal (Actor 1)")
        ax.scatter(all_ts[all_labels==1], gdn_scores[all_labels==1],
                s=2, c="tomato",    alpha=0.8, label="Anomaly (Actor 2)")
        ax.axvspan(A2_START, A2_END, alpha=0.10, color="red")
        ax.axhline(gdn_threshold, color="black", linestyle="--", lw=1)
        ax.set_title(f"GDN Anomaly Score Timeline  |  AUROC={gdn_auroc:.4f}  AUPRC={gdn_auprc:.4f}")
        ax.set_ylabel("Anomaly Score")
        ax.legend(loc="upper left", markerscale=5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)

        ax = axes[1]
        ax.hist(gdn_scores[all_labels==0], bins=100, alpha=0.6,
                density=True, color="steelblue", label="Normal")
        ax.hist(gdn_scores[all_labels==1], bins=100, alpha=0.7,
                density=True, color="tomato",    label="Anomaly")
        ax.axvline(gdn_threshold, color="black", linestyle="--")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_title("GDN Score Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig("savedWork/gdn_results.png", dpi=150)
        plt.show()
        print("Saved → savedWork/gdn_results.png")

        # Save scores and metrics for compare.py
        np.save("savedWork/gdn_scores.npy", gdn_scores)
        np.savez("savedWork/gdn_metrics.npz",
                threshold=gdn_threshold,
                auroc=gdn_auroc,
                auprc=gdn_auprc,
                fpr=gdn_fpr)

if __name__ == "__main__":
        evaluate_gdn()