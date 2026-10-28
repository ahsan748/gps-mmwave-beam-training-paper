# Sensitivity of the online trackers to the quantile-tracking step eta (64 beams, no added noise): set coverage,
# worst-stream served coverage and overhead of the online set (certificates + sweeps), sweeps-only IQT and the
# oracle tracker, for eta in {0.1, 0.25, 0.5, 1} at the chosen sweep rate. Same streams, models and seeds as main_runs.py.
# Usage: python eta_sensitivity.py --data Position-Beam-Prediction/Gathered_data_DEV --out results/eta
import argparse, os, sys, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from settings import M_SWEEP
from streams import stream_data, PAIRS
RHO = B.TARGETS["1dB"]; ETAS = [0.1, 0.25, 0.5, 1.0]


def job(D, st, seed, sweep_seeds):
    B.NB = 64
    X, pwr, seq, y, tr, ca, te = stream_data(D, st, seed); Pc, Pt = B.fit_predict("mlp", X, y, tr, [ca, te], seed)
    sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank"); rows = []
    for eta in ETAS:
        for fb in ["cert", "sweep", "oracle"]:
            for al in [0.05, 0.1]:
                for ss in range(sweep_seeds if fb != "oracle" else 1):
                    rng = np.random.default_rng(zlib.crc32(repr((st, seed, ss, fb, al, eta)).encode()))
                    r = S.simulate(Pt, pwr[te], RHO, sc, al, ("QT", fb), M=None if fb == "oracle" else M_SWEEP, eta=eta, rng=rng, seq=seq[te])
                    r = r[0] if isinstance(r, tuple) else r
                    rows.append(dict(stream=st, seed=seed, eta=eta, feedback=fb, alpha=al, sweep_seed=ss, set_cov=r["set_cov"],
                                     served=r["served"], overhead=r["overhead"]))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results/eta")
    ap.add_argument("--streams", nargs="*", default=[f"E{s}" for s in range(1, 10)] + [f"X{a}-{b}" for a, b in PAIRS])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--sweep-seeds", type=int, default=4); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = Parallel(n_jobs=a.jobs)(delayed(job)(a.data, st, sd, a.sweep_seeds) for st in a.streams for sd in range(a.seeds))
    df = pd.DataFrame([r for x in res for r in x]); df.to_csv(f"{a.out}/results_eta.csv", index=False)
    g = df.groupby(["eta", "feedback", "alpha", "stream"])[["set_cov", "served", "overhead"]].mean().reset_index()
    o = g.groupby(["eta", "feedback", "alpha"]).agg(set_cov_mean=("set_cov", "mean"), set_cov_worst=("set_cov", "min"),
                                                     served_worst=("served", "min"), overhead=("overhead", "mean"))
    o["overhead"] *= 100; o.to_csv(f"{a.out}/summary.csv"); print(o.round(3).to_string())


if __name__ == "__main__":
    main()
