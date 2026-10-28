# GPS-free grids and hybrids with a starting beam TUNED PER SITE, in every regime.
#   part "cb64":  64 beams, noise-free; grids every 4/6/8/10th, top-1 + every 8/10/12/16th, top-2 + every 12/16/20th
#   part "cb16":  16-beam codebook (every 4th beam; predictor retrained with 16 outputs), noise-free;
#                 grids every 2/3/4th, top-1 + every 3/4/5/6th, top-2 + every 4/5/6th
#   part "cb64n": 64 beams with probe noise 0.25 / 0.5 / 1 dB and re-measurement (r = 3); grids every 2/3/4/6/8/10th,
#                 top-1 + every 8/10th, top-2 + every 12/16th
# Step 1 (validation): per scenario, train on drives 0-40%, calibrate on 40-55%, replay every starting beam
#   (offset 0..d-1) on the validation drives 55-80% (with the same probe noise, 2 noise seeds), and choose the start
#   with the highest validation served coverage (ties: lower overhead, then lower offset).
# Step 2 (test): replay the chosen start on the 15 test streams (same streams, seeds and predictor as main_runs.py);
#   protocol-E stream s uses scenario s's choice, cross-session stream a->b uses the SOURCE scenario a's choice.
#   Per-sample traces are saved so bootstrap_tuned.py can pair them with the other traces.
# Usage: python tuned_grid_runs.py --data Position-Beam-Prediction/Gathered_data_DEV --part cb16 --out results/tuned_grids
import argparse, os, pickle, sys, time, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from settings import ETA
from streams import stream_data, PAIRS

RHO = B.TARGETS["1dB"]
CFG = {"cb64": dict(nb=64, noise=[0.0], fam=[("uniform", d) for d in [4, 6, 8, 10]] + [("hybrid", d) for d in [8, 10, 12, 16]] + [("hybrid2", d) for d in [12, 16, 20]]),
       "cb16": dict(nb=16, noise=[0.0], fam=[("uniform", d) for d in [2, 3, 4]] + [("hybrid", d) for d in [3, 4, 5, 6]] + [("hybrid2", d) for d in [4, 5, 6]]),
       "cb64n": dict(nb=64, noise=[0.25, 0.5, 1.0], fam=[("uniform", d) for d in [2, 3, 4, 6, 8, 10]] + [("hybrid", d) for d in [8, 10]] + [("hybrid2", d) for d in [12, 16]])}


def pol(sch, d, o): return ("uniform", (d, o)) if sch == "uniform" else ("hybrid", (int(sch[6:] or 1), d, o))   # "hybrid2" = top-2
def rng_for(*key): return np.random.default_rng(zlib.crc32(repr(key).encode()))
def n_noise_seeds(nz): return 1 if nz == 0 else 2


def val_job(D, part, s, seed):
    cfg = CFG[part]; nb = cfg["nb"]; B.NB = nb
    X, pwr, seq = B.load(D, s); pwr = pwr[:, ::64 // nb]; y = pwr.argmax(1); u = np.unique(seq); k = len(u)
    tr = np.where(np.isin(seq, u[:int(.4 * k)]))[0]; ca = np.where(np.isin(seq, u[int(.4 * k):int(.55 * k)]))[0]
    va = np.where(np.isin(seq, u[int(.55 * k):int(.8 * k)]))[0]
    Pc, Pv = B.fit_predict("mlp", X, y, tr, [ca, va], seed); sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank")
    rows = []
    for nz in cfg["noise"]:
        rm = 3 if nz > 0 else 0
        for sch, d in cfg["fam"]:
            for o in range(d):
                for ns in range(n_noise_seeds(nz)):
                    r = S.simulate(Pv, pwr[va], RHO, sc, 0.1, pol(sch, d, o), rng=rng_for("val", part, s, seed, ns, sch, d, o, nz),
                                   seq=seq[va], eta=ETA, noise_db=nz, remeasure=rm)
                    r = r[0] if isinstance(r, tuple) else r
                    rows.append(dict(part=part, noise=nz, scheme=sch, d=d, offset=o, scenario=s, seed=seed, noise_seed=ns,
                                     served=r["served"], overhead=r["overhead"]))
    return rows


def src(stream): return int(stream[1]) if stream.startswith("E") else int(stream.split("-")[0][1:])


def test_job(D, part, stream, seed, choice):
    cfg = CFG[part]; nb = cfg["nb"]; B.NB = nb
    X, pwr, seq, y, tr, ca, te = stream_data(D, stream, seed)
    pwr = pwr[:, ::64 // nb]; y = pwr.argmax(1)
    Pc, Pt = B.fit_predict("mlp", X, y, tr, [ca, te], seed); sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank")
    rows, traces = [], {}
    for nz in cfg["noise"]:
        rm = 3 if nz > 0 else 0
        for sch, d in cfg["fam"]:
            o = choice[(nz, sch, d, src(stream))]; name = f"tuned-{sch}:{d}"
            for ns in range(n_noise_seeds(nz)):
                r, t_ = S.simulate(Pt, pwr[te], RHO, sc, 0.1, pol(sch, d, o), rng=rng_for("test", part, stream, seed, ns, sch, d, nz),
                                   seq=seq[te], eta=ETA, noise_db=nz, remeasure=rm, rec_traces=True)
                rows.append(dict(part=part, stream=stream, seed=seed, noise=nz, remeasure=rm, noise_seed=ns, name=name, offset=o, **r))
                traces.setdefault((nz, name), []).append(dict(drive=seq[te][t_["keep"]], served=t_["served"], drive_all=seq[te],
                                                              probes=t_["probes"]))
    return rows, (stream, seed, traces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--part", default="cb16", choices=list(CFG))
    ap.add_argument("--streams", nargs="*", default=[f"E{s}" for s in range(1, 10)] + [f"X{a}-{b}" for a, b in PAIRS])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--sweep-seeds", type=int, default=1)   # accepted for run_all --quick
    ap.add_argument("--jobs", type=int, default=-1); ap.add_argument("--out", default="results/tuned_grids")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    scen = sorted({src(st) for st in a.streams})
    va = Parallel(n_jobs=a.jobs)(delayed(val_job)(a.data, a.part, s, sd) for s in scen for sd in range(a.seeds))
    V = pd.DataFrame([r for rows in va for r in rows]); V.to_csv(f"{a.out}/validation_{a.part}.csv", index=False)
    v = V.groupby(["noise", "scheme", "d", "scenario", "offset"])[["served", "overhead"]].mean().reset_index()
    ch = (v.sort_values(["served", "overhead", "offset"], ascending=[False, True, True])
           .groupby(["noise", "scheme", "d", "scenario"]).head(1))
    ch.to_csv(f"{a.out}/choices_{a.part}.csv", index=False)
    choice = {(r.noise, r.scheme, r.d, r.scenario): int(r.offset) for r in ch.itertuples()}
    # spacing chosen on validation too: cheapest spacing (with tuned starts) whose worst validation scenario >= 0.9
    vw = ch.groupby(["noise", "scheme", "d"]).agg(val_worst=("served", "min"), val_ovh=("overhead", "mean")).reset_index()
    vw.to_csv(f"{a.out}/validation_spacing_{a.part}.csv", index=False)
    print(f"[{a.part}] validation (tuned starts), worst scenario and mean overhead per spacing:"); print(vw.round(3).to_string(index=False))
    res = Parallel(n_jobs=a.jobs)(delayed(test_job)(a.data, a.part, st, sd, choice) for st in a.streams for sd in range(a.seeds))
    T = pd.DataFrame([r for rows, _ in res for r in rows]); T.to_csv(f"{a.out}/results_{a.part}.csv", index=False)
    with open(f"{a.out}/traces_{a.part}.pkl", "wb") as f: pickle.dump([t for _, t in res], f)
    g = T.groupby(["noise", "name", "stream"])[["served", "overhead"]].mean().reset_index()
    print(f"\n[{a.part}] test streams with tuned starts:")
    print(g.groupby(["noise", "name"]).agg(worst=("served", "min"), ovh=("overhead", "mean")).round(3).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
