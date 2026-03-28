import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

from config import ACTOR1_END, ACTOR2_START, ACTOR2_END, ACTOR1_RETURN, RESAMPLE_FREQ


def preprocess():
    os.makedirs("savedWork", exist_ok=True)
    os.makedirs("saved_ds",  exist_ok=True)

    print("Loading CSV...")
    data = pd.read_csv(
        r"BREMaster.csv",
        parse_dates=["Timestamp"],
        low_memory=False
    )
    data = data.sort_values("Timestamp").reset_index(drop=True)
    print(f"  Raw rows : {len(data):,}  |  Cols: {data.shape[1]}")

    sensor_cols = data.columns.drop("Timestamp").tolist()
    data[sensor_cols] = data[sensor_cols].ffill().bfill()

    def parse_sensor(col):
        base, feat = col.split('_', 1)
        parts = base.split('-')
        return int(parts[1]), parts[2], feat

    sorted_cols = sorted(sensor_cols, key=parse_sensor)
    data = data[["Timestamp"] + sorted_cols]

    # Resample to 5-second buckets
    data = data.set_index("Timestamp")
    data = (
        data[sorted_cols]
        .resample(RESAMPLE_FREQ)
        .mean()
        .ffill()
        .reset_index()
    )
    print(f"  After {RESAMPLE_FREQ} resample : {len(data):,} rows  "
        f"({len(data)/2_671_187*100:.1f}% of original)")

    ts = data["Timestamp"]
    actor1_train_mask = ts < pd.Timestamp(ACTOR2_START)
    actor2_mask = (
        (ts >= pd.Timestamp(ACTOR2_START)) &
        (ts <= pd.Timestamp(ACTOR2_END)
        + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )

    print(f"  Actor 1 train rows  : {actor1_train_mask.sum():,}")
    print(f"  Actor 2 anomaly rows: {actor2_mask.sum():,}  "
        f"({actor2_mask.mean()*100:.1f}%)")

    scaler = MinMaxScaler()
    scaler.fit(data.loc[actor1_train_mask, sorted_cols])
    data[sorted_cols] = scaler.transform(data[sorted_cols])

    # Cast to float32 — halves memory vs float64
    data[sorted_cols] = data[sorted_cols].astype(np.float32)
    data["label"]     = actor2_mask.astype(int)

    data.to_csv("savedWork/cleaned_data.csv", index=False)

    NUM_SENSORS = len(sorted_cols)
    print(f"  Sensors : {NUM_SENSORS}")
    print(f"  Saved   → savedWork/cleaned_data.csv")
    print(f"  Memory  : ~{len(data)*NUM_SENSORS*4/1e9:.2f} GB float32")

if __name__ == "__main__":
    preprocess()
