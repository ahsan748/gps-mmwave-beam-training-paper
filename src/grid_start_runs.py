# Sensitivity of the GPS-free grid and the hybrid to the grid's starting beam (offset 0..d-1).
# 64-beam codebook, noise-free, same streams, seeds and predictor as main_runs.py.
# Usage: python grid_start_runs.py --data Position-Beam-Prediction/Gathered_data_DEV --out results/grid_start
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from settings import ETA
from streams import stream_data, PAIRS

RHO = B.TARGETS["1dB"]
POLS = [("uniform", (d, o)) for d in [4, 6, 8, 10] for o in range(d)] + [("hybrid", (1, d, o)) for d in [8, 10] for o in range(d)] + \
       [("hybrid", (2, d, o)) for d in [12, 16] for o in range(d)]


def job(D, stream, seed):
    B.NB = 64
    X, pwr, seq, y, tr, ca, te = stream_data(D, stream, seed)
    Pc, Pt = B.fit_predict("mlp", X, y, tr, [ca, te], seed); sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank")
    rows = []
    for pol in POLS:
        r = S.simulate(Pt, pwr[te], RHO, sc, 0.1, pol, rng=np.random.default_rng(seed), seq=seq[te], eta=ETA)
        r = r[0] if isinstance(r, tuple) else r
        d, o = (pol[1][0], pol[1][1]) if pol[0] == "uniform" else (pol[1][1], pol[1][2])
        sch = pol[0] if pol[0] == "uniform" or pol[1][0] == 1 else f"hybrid{pol[1][0]}"   # "hybrid" = top-1, "hybrid2" = top-2
        rows.append(dict(stream=stream, seed=seed, scheme=sch, d=d, offset=o, **r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results/grid_start")
    ap.add_argument("--streams", nargs="*", default=[f"E{s}" for s in range(1, 10)] + [f"X{a}-{b}" for a, b in PAIRS])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--sweep-seeds", type=int, default=1)  # unused (no sweeps)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(job)(a.data, st, sd) for st in a.streams for sd in range(a.seeds))
    df = pd.DataFrame([r for rows in res for r in rows]); df.to_csv(f"{a.out}/results_offsets.csv", index=False)
    g = df.groupby(["scheme", "d", "offset", "stream"])[["served", "overhead"]].mean().reset_index()
    w = g.groupby(["scheme", "d", "offset"]).agg(worst=("served", "min"), ovh=("overhead", "mean")).reset_index()
    print(w.groupby(["scheme", "d"]).agg(worst_min=("worst", "min"), worst_mean=("worst", "mean"), worst_max=("worst", "max"),
                                         worst_at0=("worst", "first"), ovh=("ovh", "mean")).round(3).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
