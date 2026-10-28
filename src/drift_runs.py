# Drift table, feedback ablation and the kNN runs (simulator: beamsim.py).
# Streams: the future-drive protocol (scenarios 1-9, last 20% of drives) and 6 same-site cross-session pairs; a random split for the drift table.
# Parts: (1) drift: static conformal set coverage under A / E / X
#        (2) feedback ablation (alpha 0.1)
#        (3) matched-reliability frontier: GPS top-K, GPS+grid hybrids, coarse grids, tracking, exhaustive, online sets
# Every serving policy uses the same hill-climb; every probe (set, climb, sweep) is counted.
# Usage:  python drift_runs.py --data Position-Beam-Prediction/Gathered_data_DEV --out results/drift
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from settings import ETA, M_SWEEP
from streams import stream_data, PAIRS

RHO = B.TARGETS["1dB"]
FRONT_FIXED = ([("fixed", k) for k in [1, 2, 3, 5, 8, 12, 16, 24, 32]] + [("uniform", d) for d in [3, 4, 5, 6, 7, 8, 10, 12, 16]] +
               [("hybrid", (1, d)) for d in [6, 7, 8, 10, 12, 16]] + [("hybrid", (2, d)) for d in [8, 10, 12]] +
               [("track", None), ("fixed", 64)])
KNN_FIXED = [("fixed", k) for k in [3, 8, 16, 24]] + [("hybrid", (1, d)) for d in [8, 10, 12]] + [("hybrid", (2, d)) for d in [12, 16, 20]]
ALPHAS = [0.05, 0.1, 0.2, 0.3]


def job(D, stream, seed, sweep_seeds, noises, eta=ETA):
    rows = []
    for pred in ["mlp", "knn"]:
        X, pwr, seq, y, tr, ca, te = stream_data(D, stream, seed)
        Pc, Pt = B.fit_predict(pred, X, y, tr, [ca, te], seed)
        sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank")
        base = dict(stream=stream, seed=seed, predictor=pred, top1=(Pt.argmax(1) == y[te]).mean(),
                    top5=(B.ranks(Pt)[np.arange(len(te)), y[te]] <= 5).mean(), n_test=len(te))
        sim = lambda pol, al, rng, nz, M=None: S.simulate(Pt, pwr[te], RHO, sc, al, pol, M=M, rng=rng, noise_db=nz, seq=seq[te], eta=eta)
        # (1) drift: static conformal (also protocol A for the E streams)
        r = sim(("static", None), 0.1, np.random.default_rng(0), 0.0)
        rows.append(dict(base, part="drift", protocol="E" if stream.startswith("E") else "X", policy="static", par="None",
                         alpha=0.1, noise=0.0, sweep_seed=0, M=0, **r))
        if stream.startswith("E"):
            s = int(stream[1:]); Xa, pa, qa = B.load(D, s); ya = pa.argmax(1)
            trA, caA, teA = B.splits(qa, np.random.default_rng(seed))["A_random"]
            PcA, PtA = B.fit_predict(pred, Xa, ya, trA, [caA, teA], seed)
            rA = S.simulate(PtA, pa[teA], RHO, B.scores(PcA, B.good_mask(pa[caA], RHO), "rank"), 0.1, ("static", None),
                            rng=np.random.default_rng(0), seq=qa[teA])
            rows.append(dict(base, part="drift", protocol="A", policy="static", par="None", alpha=0.1, noise=0.0,
                             sweep_seed=0, M=0, **rA))
        for nz in noises:
            for ss in range(sweep_seeds):
                rng = lambda k: np.random.default_rng(100_000 * seed + 1000 * ss + 10 * k + int(10 * nz))
                # (3) frontier: deterministic policies need one run without noise
                if ss == 0 or nz > 0:
                    for i, pol in enumerate(FRONT_FIXED if pred == "mlp" else KNN_FIXED):
                        r = sim(pol, 0.1, rng(i), nz)
                        rows.append(dict(base, part="frontier", policy=pol[0], par=str(pol[1]), alpha=0.1, noise=nz,
                                         sweep_seed=ss, M=0, **r))
                # online sets (frontier over alpha) and (2) the feedback ablation at alpha 0.1
                for al in (ALPHAS if pred == "mlp" else [0.1]):
                    for fb, M in [("cert", M_SWEEP), ("oracle", None)] + ([("sweep", M_SWEEP), ("cert", None)] if al == 0.1 else []):
                        if pred == "knn" and fb != "cert": continue
                        r = sim(("QT", fb), al, rng(50 + int(100 * al) + (M or 0)), nz, M)
                        rows.append(dict(base, part="online", policy="QT", par=fb + ("" if M else "-nosweep" if fb == "cert" else ""),
                                         alpha=al, noise=nz, sweep_seed=ss, M=M or 0, **r))
                    if al == 0.1 and pred == "mlp":
                        r = sim(("ACI", "cert"), al, rng(99), nz, M_SWEEP)
                        rows.append(dict(base, part="online", policy="ACI", par="cert", alpha=al, noise=nz, sweep_seed=ss, M=M_SWEEP, **r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--streams", nargs="*", default=[f"E{s}" for s in range(1, 10)] + [f"X{a}-{b}" for a, b in PAIRS])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweep-seeds", type=int, default=4)
    ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 1.0])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default="results/drift")
    ap.add_argument("--eta", type=float, default=ETA, help="QT step in beams (selected on the validation stream, settings.py)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    tasks = [(st, sd) for st in a.streams for sd in range(a.seeds)]
    print(f"running {len(tasks)} jobs ...", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(job)(a.data, st, sd, a.sweep_seeds, a.noise, a.eta) for st, sd in tasks)
    df = pd.DataFrame([r for rows in res for r in rows]); df.to_csv(f"{a.out}/results_drift.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s -> {a.out}/results_drift.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
