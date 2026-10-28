# Main runs.
#   part "cb64": 64-beam codebook; noise 0 / 0.25 / 0.5 / 1 dB with and without re-measurement, tracking + periodic
#                coarse grid, ACI, per-sample traces for the bootstrap
#   part "cb16": the same comparison on the coarse 16-beam codebook (every 4th beam of the 64)
# Usage:  python main_runs.py --data Position-Beam-Prediction/Gathered_data_DEV --part cb64 --out results/main
import argparse, zlib, os, pickle, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from settings import ETA, M_SWEEP
from streams import stream_data, PAIRS

RHO = B.TARGETS["1dB"]
POL = {
    "cb64": dict(
        clean=[("uniform", d) for d in [3, 4, 5, 6, 7, 8, 10, 12]] + [("hybrid", (1, d)) for d in [6, 7, 8, 10, 12, 16]] + [("hybrid", (2, d)) for d in [8, 10, 12, 16, 20]] + [("hybrid", (3, 16))] +
              [("fixed", k) for k in [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48]] + [("fixed", 64), ("track", None)] +
              [("trackgrid", (n, 6)) for n in [5, 10, 20]] + [("trackgrid", (n, 4)) for n in [5, 10]] + [("static", None)],
        noisy=[("uniform", d) for d in [4, 6, 8, 10]] + [("hybrid", (1, d)) for d in [8, 10]] + [("hybrid", (2, d)) for d in [12, 16]] +
              [("fixed", k) for k in [16, 24, 32]] + [("fixed", 64), ("trackgrid", (10, 6))],
        online=[0.02, 0.05, 0.1, 0.2], noise=[0.25, 0.5, 1.0], trace=["uniform:6", "uniform:8", "uniform:10", "uniform:5",
              "hybrid:(1, 8)", "hybrid:(1, 10)", "hybrid:(1, 12)", "hybrid:(1, 16)", "hybrid:(2, 8)", "hybrid:(2, 10)", "hybrid:(2, 12)", "hybrid:(2, 16)", "hybrid:(2, 20)", "fixed:16", "fixed:20", "fixed:40", "fixed:48", "fixed:24", "fixed:32", "QT-oracle:0.02", "QT-cert:0.02", "QT-oracle:0.05", "QT-cert:0.05",
              "QT-oracle:0.1", "QT-cert:0.1"]),
    "cb16": dict(
        clean=[("uniform", d) for d in [2, 3, 4]] + [("hybrid", (1, d)) for d in [3, 4, 5, 6]] + [("hybrid", (2, d)) for d in [4, 5, 6]] +
              [("fixed", k) for k in [1, 2, 3, 4, 5, 6, 8, 10]] + [("fixed", 16), ("track", None), ("trackgrid", (10, 3))],
        noisy=[("uniform", d) for d in [2, 3, 4]] + [("hybrid", (1, d)) for d in [3, 4, 5]] + [("fixed", k) for k in [4, 6, 8]] +
              [("fixed", 16)],
        online=[0.02, 0.05, 0.1, 0.2], noise=[0.5, 1.0], trace=[]),
}


def job(D, part, stream, seed, sweep_seeds):
    cfg = POL[part]; nbeams = 64 if part == "cb64" else 16; step = 64 // nbeams
    B.NB = nbeams                                                # predictor output size and rank-score cap
    X, pwr, seq, y, tr, ca, te = stream_data(D, stream, seed)
    pwr = pwr[:, ::step]; y = pwr.argmax(1)                      # decimated codebook: keep every 4th beam
    Pc, Pt = B.fit_predict("mlp", X, y, tr, [ca, te], seed)
    sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank")
    base = dict(part=part, stream=stream, seed=seed, nbeams=nbeams, n_test=len(te), top1=(Pt.argmax(1) == y[te]).mean())
    rows, traces = [], {}
    def run(pol, al, nz, rm, ss, M=None, key=None):
        rng = np.random.default_rng(zlib.crc32(repr((stream, seed, ss, str(pol), al, nz, rm)).encode()))
        r, tr_ = S.simulate(Pt, pwr[te], RHO, sc, al, pol, M=M, rng=rng, noise_db=nz, seq=seq[te], eta=ETA,
                            remeasure=rm, rec_traces=True)
        name = key or f"{pol[0]}:{pol[1]}"
        rows.append(dict(base, name=name, alpha=al, noise=nz, remeasure=rm, sweep_seed=ss, M=M or 0, **r))
        if nz == 0 and name in cfg["trace"]:
            traces.setdefault(name, []).append(dict(drive=seq[te][tr_["keep"]], served=tr_["served"],
                                                    drive_all=seq[te], probes=tr_["probes"]))
    for pol in cfg["clean"]:
        run(pol, 0.1, 0.0, 0, 0)
    for al in cfg["online"]:
        for ss in range(sweep_seeds):
            run(("QT", "cert"), al, 0.0, 0, ss, M=M_SWEEP, key=f"QT-cert:{al}")
            run(("QT", "oracle"), al, 0.0, 0, ss, key=f"QT-oracle:{al}")
    if part == "cb64":
        for ss in range(sweep_seeds):                            # feedback ablation, same seeds and file as everything else
            run(("ACI", "cert"), 0.1, 0.0, 0, ss, M=M_SWEEP, key="ACI-cert:0.1")
            for al in [0.02, 0.05, 0.1, 0.2]:                      # sweeps-only IQT at every alpha (validation-chosen points)
                run(("QT", "sweep"), al, 0.0, 0, ss, M=M_SWEEP, key=f"QT-sweep:{al}")
            run(("QT", "cert"), 0.1, 0.0, 0, ss, M=None, key="QT-certonly:0.1")
    for nz in cfg["noise"]:
        for rm in [0, 3]:
            for ss in range(2):
                for pol in cfg["noisy"]:
                    run(pol, 0.1, nz, rm, ss)
                for al in [0.05, 0.1]:
                    run(("QT", "cert"), al, nz, rm, ss, M=M_SWEEP, key=f"QT-cert:{al}")
    return rows, (stream, seed, traces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--part", default="cb64", choices=list(POL))
    ap.add_argument("--streams", nargs="*", default=[f"E{s}" for s in range(1, 10)] + [f"X{a}-{b}" for a, b in PAIRS])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--sweep-seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=-1); ap.add_argument("--out", default="results/main")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    tasks = [(st, sd) for st in a.streams for sd in range(a.seeds)]
    print(f"[{a.part}] running {len(tasks)} jobs ...", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(job)(a.data, a.part, st, sd, a.sweep_seeds) for st, sd in tasks)
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(f"{a.out}/results_{a.part}.csv", index=False)
    if a.part == "cb64":
        with open(f"{a.out}/traces_{a.part}.pkl", "wb") as f: pickle.dump([t for _, t in res], f)
    print(f"done in {time.time() - t0:.0f} s -> {a.out}/results_{a.part}.csv", flush=True)


if __name__ == "__main__":
    main()
