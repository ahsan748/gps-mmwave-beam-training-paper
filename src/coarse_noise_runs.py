# Coarse-codebook and probe-noise runs (same simulator, seeds and streams as main_runs.py).
#   part "cb16": coarse 16-beam codebook, noise-free and 0.5 dB; adds top-12/14 and sweeps-only IQT; traces for the bootstrap
#   part "cb64n": 64-beam codebook under probe noise (0.25 / 0.5 / 1 dB, with and without re-measurement); adds
#                 grids every 2nd/3rd beam and sweeps-only IQT; traces for the bootstrap
# Usage: python coarse_noise_runs.py --data Position-Beam-Prediction/Gathered_data_DEV --part cb16 --out results/coarse_noise
import argparse, zlib, os, pickle, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from settings import ETA, M_SWEEP
from streams import stream_data, PAIRS

RHO = B.TARGETS["1dB"]
CFG = {
    "cb16": dict(nb=16, noise=[0.0, 0.5],
                 fixed=[("uniform", d) for d in [2, 3, 4]] + [("hybrid", (1, d)) for d in [3, 4, 5, 6]] +
                       [("hybrid", (2, d)) for d in [4, 5, 6]] + [("fixed", k) for k in [3, 4, 6, 8, 10, 12, 14]] +
                       [("fixed", 16), ("track", None)],
                 online=[("QT", "cert", a, M_SWEEP) for a in [0.02, 0.05, 0.1, 0.2]] + [("QT", "oracle", a, None) for a in [0.02, 0.05, 0.1, 0.2]] +
                        [("QT", "sweep", a, M_SWEEP) for a in [0.02, 0.05, 0.1, 0.2]]),
    "cb64n": dict(nb=64, noise=[0.25, 0.5, 1.0],
                  fixed=[("uniform", d) for d in [2, 3, 4, 6]] + [("hybrid", (1, d)) for d in [8, 10]] + [("hybrid", (2, d)) for d in [12, 16]] +
                        [("fixed", k) for k in [32]] + [("fixed", 64)],
                  online=[("QT", "cert", a, M_SWEEP) for a in [0.05, 0.1]] + [("QT", "sweep", 0.05, M_SWEEP)]),
}


def job(D, part, stream, seed, sweep_seeds):
    cfg = CFG[part]; nb = cfg["nb"]; B.NB = nb
    X, pwr, seq, y, tr, ca, te = stream_data(D, stream, seed)
    pwr = pwr[:, ::64 // nb]; y = pwr.argmax(1)                  # 16 beams: retrained with 16 outputs, recalibrated
    Pc, Pt = B.fit_predict("mlp", X, y, tr, [ca, te], seed); sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank")
    base = dict(part=part, stream=stream, seed=seed, nbeams=nb, n_test=len(te), top1=(Pt.argmax(1) == y[te]).mean())
    rows, traces = [], {}
    def run(pol, al, nz, rm, ss, M=None, key=None):
        rng = np.random.default_rng(zlib.crc32(repr((stream, seed, ss, str(pol), al, nz, rm, part)).encode()))
        r, t_ = S.simulate(Pt, pwr[te], RHO, sc, al, pol, M=M, rng=rng, noise_db=nz, seq=seq[te], eta=ETA,
                           remeasure=rm, rec_traces=True)
        name = key or f"{pol[0]}:{pol[1]}"
        rows.append(dict(base, name=name, alpha=al, noise=nz, remeasure=rm, sweep_seed=ss, M=M or 0, **r))
        if rm == (3 if nz > 0 else 0):                           # traces: noise-free, or re-measured under noise
            traces.setdefault((nz, name), []).append(dict(drive=seq[te][t_["keep"]], served=t_["served"],
                                                          drive_all=seq[te], probes=t_["probes"]))
    for nz in cfg["noise"]:
        for rm in ([0] if nz == 0 else [0, 3]):
            n_ss = sweep_seeds if nz == 0 else 2
            for ss in range(n_ss):
                if nz > 0 or ss == 0:
                    for pol in cfg["fixed"]: run(pol, 0.1, nz, rm, ss)
                for p, fb, al, M in cfg["online"]:
                    run((p, fb), al, nz, rm, ss, M=M, key=f"QT-{fb}:{al}")
    return rows, (stream, seed, traces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--part", default="cb16", choices=list(CFG))
    ap.add_argument("--streams", nargs="*", default=[f"E{s}" for s in range(1, 10)] + [f"X{a}-{b}" for a, b in PAIRS])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--sweep-seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=-1); ap.add_argument("--out", default="results/coarse_noise")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    tasks = [(st, sd) for st in a.streams for sd in range(a.seeds)]
    print(f"[{a.part}] running {len(tasks)} jobs ...", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(job)(a.data, a.part, st, sd, a.sweep_seeds) for st, sd in tasks)
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(f"{a.out}/results_{a.part}.csv", index=False)
    with open(f"{a.out}/traces_{a.part}.pkl", "wb") as f: pickle.dump([t for _, t in res], f)
    print(f"done in {time.time() - t0:.0f} s -> {a.out}/results_{a.part}.csv", flush=True)


if __name__ == "__main__":
    main()
