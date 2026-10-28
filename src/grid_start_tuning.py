# Grid starting beam tuned per site: for each scenario, the start (offset) of each grid/hybrid is chosen on the
# validation stream (drives 55-80%; predictor trained on 0-40%, calibrated on 40-55%, as in validate_settings.py)
# by the highest validation served coverage (ties: lower overhead, then lower offset). It is then applied to the
# test results of grid_start_runs.py: future-drive stream s uses scenario s's choice; cross-session stream a->b uses
# the SOURCE scenario a's choice (the target session is unseen). Usage:
#   python grid_start_tuning.py --data Position-Beam-Prediction/Gathered_data_DEV --grid-start results/grid_start --out results/grid_start_tuned
import argparse, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from settings import ETA

RHO = B.TARGETS["1dB"]
FAM = [("uniform", 4), ("uniform", 6), ("uniform", 8), ("uniform", 10), ("hybrid", 8), ("hybrid", 10), ("hybrid2", 12), ("hybrid2", 16)]


def pol(sch, d, o): return ("uniform", (d, o)) if sch == "uniform" else ("hybrid", (int(sch[6:] or 1), d, o))


def job(D, s, seed):
    B.NB = 64
    X, pwr, seq = B.load(D, s); y = pwr.argmax(1); u = np.unique(seq); k = len(u)
    tr = np.where(np.isin(seq, u[:int(.4 * k)]))[0]; ca = np.where(np.isin(seq, u[int(.4 * k):int(.55 * k)]))[0]
    va = np.where(np.isin(seq, u[int(.55 * k):int(.8 * k)]))[0]
    Pc, Pv = B.fit_predict("mlp", X, y, tr, [ca, va], seed); sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank")
    rows = []
    for sch, d in FAM:
        for o in range(d):
            r = S.simulate(Pv, pwr[va], RHO, sc, 0.1, pol(sch, d, o), rng=np.random.default_rng(seed), seq=seq[va], eta=ETA)
            r = r[0] if isinstance(r, tuple) else r
            rows.append(dict(scenario=s, seed=seed, scheme=sch, d=d, offset=o, served=r["served"], overhead=r["overhead"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--grid-start", dest="grid_start", default="results/grid_start"); ap.add_argument("--out", default="results/grid_start_tuned")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--streams", nargs="*", default=None); ap.add_argument("--sweep-seeds", type=int, default=1)  # accepted for run_all --quick
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    te = pd.read_csv(f"{a.grid_start}/results_offsets.csv")
    scen = sorted({int(st[1]) for st in te.stream.unique()} | {int(st.split("-")[0][1:]) for st in te.stream.unique() if st.startswith("X")})
    res = Parallel(n_jobs=a.jobs)(delayed(job)(a.data, s, sd) for s in scen for sd in range(a.seeds))
    va = pd.DataFrame([r for rows in res for r in rows]); va.to_csv(f"{a.out}/validation_offsets.csv", index=False)
    v = va.groupby(["scheme", "d", "scenario", "offset"])[["served", "overhead"]].mean().reset_index()
    choice = (v.sort_values(["served", "overhead", "offset"], ascending=[False, True, True])
               .groupby(["scheme", "d", "scenario"]).head(1).set_index(["scheme", "d", "scenario"]).offset)
    t = te.groupby(["scheme", "d", "offset", "stream"])[["served", "overhead"]].mean().reset_index()
    src = lambda st: int(st[1]) if st.startswith("E") else int(st.split("-")[0][1:])
    out, rows = [], []
    for sch, d in FAM:
        tt = t[(t.scheme == sch) & (t.d == d)]
        per = []
        for st, g in tt.groupby("stream"):
            o = choice.loc[(sch, d, src(st))]; r = g[g.offset == o].iloc[0]; best = g.served.max()
            per.append(dict(scheme=sch, d=d, stream=st, offset=o, served=r.served, overhead=r.overhead, best_offset=int(g.loc[g.served.idxmax(), "offset"]),
                            best_served=best, at_best=bool(abs(r.served - best) < 1e-12), within_001=bool(best - r.served <= 0.01)))
        P = pd.DataFrame(per); rows += per
        e = P[P.stream.str.startswith("E")]
        out.append(dict(scheme=sch, d=d, worst_tuned=P.served.min(), worst_stream=P.loc[P.served.idxmin(), "stream"], ovh_tuned=100 * P.overhead.mean(),
                        worst_tuned_E=e.served.min(), ovh_tuned_E=100 * e.overhead.mean(), share_best=P.at_best.mean(), share_within_001=P.within_001.mean(),
                        worst_offset0=tt[tt.offset == 0].groupby("stream").served.mean().min()))
    pd.DataFrame(rows).to_csv(f"{a.out}/tuned_offsets_per_stream.csv", index=False)
    O = pd.DataFrame(out); O.to_csv(f"{a.out}/tuned_offsets.csv", index=False)
    print("Validation-chosen starting beam per scenario:"); print(choice.unstack("scenario").to_string())
    print("\nTest streams with the starting beam tuned on validation:"); print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
