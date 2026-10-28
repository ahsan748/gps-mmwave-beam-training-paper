# Number of test samples and test drives of each of the 15 test streams (and of the training data).
# Usage: python stream_counts.py --data Position-Beam-Prediction/Gathered_data_DEV --out results/streams.csv
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streams import stream_data, PAIRS
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results/streams.csv")
a = ap.parse_args(); rows = []
for st in [f"E{s}" for s in range(1, 10)] + [f"X{x}-{y}" for x, y in PAIRS]:
    X, pwr, seq, y, tr, ca, te = stream_data(a.data, st, 0)
    rows.append(dict(stream=st, protocol="future-drive" if st[0] == "E" else "cross-session", test_samples=len(te),
                     test_drives=len(np.unique(seq[te])), train_samples=len(tr), train_drives=len(np.unique(seq[tr])),
                     cal_samples=len(ca), cal_drives=len(np.unique(seq[ca]))))
df = pd.DataFrame(rows); df.to_csv(a.out, index=False); print(df.to_string(index=False))
