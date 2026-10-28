# Test streams: the future-drive protocol (scenarios 1-9: train on the first 55% of drives, calibrate on the next
# 25%, serve the last 20% in time order) and the cross-session protocol (six same-BS scenario pairs).
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B

PAIRS = [(1, 2), (2, 1), (3, 4), (4, 3), (8, 9), (9, 8)]


def stream_data(D, stream, seed):
    if stream.startswith("E"):
        s = int(stream[1:]); X, pwr, seq = B.load(D, s); y = pwr.argmax(1)
        tr, ca, te = B.splits(seq, np.random.default_rng(seed))["E_blocked_future"]
        return X, pwr, seq, y, tr, ca, te
    a, b = (int(v) for v in stream[1:].split("-"))
    Xs, Ps, qs = B.load(D, a); Xt, Pt_, qt = B.load(D, b)
    u = np.unique(qs); k = len(u)
    tr = np.where(np.isin(qs, u[:int(.7 * k)]))[0]; ca = np.where(np.isin(qs, u[int(.7 * k):]))[0]
    X = np.vstack([Xs, Xt]); pwr = np.vstack([Ps, Pt_]); seq = np.r_[qs, qt + qs.max() + 1]
    return X, pwr, seq, pwr.argmax(1), tr, ca, np.arange(len(Xs), len(X))


