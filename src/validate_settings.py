# Selection of every design choice on a VALIDATION stream that never touches the test drives.
# Per scenario: train on the first 40% of drives, calibrate on 40-55%, validate on 55-80% (last 20% = test, never used).
# Chooses: (1) score type for the static set (rank vs probability threshold),
#          (2) eta and M for QT + certificates,
#          (3) the cheapest coarse grid d, and the cheapest GPS+grid hybrid (K, d), whose WORST validation stream
#              reaches served coverage >= 0.90.
# Usage: python validate_settings.py --data Position-Beam-Prediction/Gathered_data_DEV --out results/drift
import argparse, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S

RHO = B.TARGETS["1dB"]; ETAS = [0.1, 0.25, 0.5, 1.0, 2.0]; MS = [20, 50, 100]
GRIDS = [("uniform", d) for d in [3, 4, 5, 6, 7, 8, 10, 12, 16]] + \
        [("hybrid", (k, d)) for k in [1, 2, 3] for d in [6, 7, 8, 10, 12, 16]]


def job(D, s, seed):
    X, pwr, seq = B.load(D, s); y = pwr.argmax(1); u = np.unique(seq); k = len(u)
    tr = np.where(np.isin(seq, u[:int(.4 * k)]))[0]; ca = np.where(np.isin(seq, u[int(.4 * k):int(.55 * k)]))[0]
    va = np.where(np.isin(seq, u[int(.55 * k):int(.8 * k)]))[0]
    Pc, Pv = B.fit_predict("mlp", X, y, tr, [ca, va], seed); rows = []
    Gc, Gv = B.good_mask(pwr[ca], RHO), B.good_mask(pwr[va], RHO)
    for kind in ["rank", "thr"]:
        cov, size = B.static_set(Pv, Gv, B.scores(Pc, Gc, kind), 0.1, kind)
        rows.append(dict(scenario=s, seed=seed, what="score", choice=kind, set_cov=cov, size=size))
    sc = B.scores(Pc, Gc, "rank")
    for eta in ETAS:
        for M in MS:
            for ss in range(2):
                r = S.simulate(Pv, pwr[va], RHO, sc, 0.1, ("QT", "cert"), M=M, eta=eta, seq=seq[va],
                               rng=np.random.default_rng(1000 * seed + 10 * ss + M))
                rows.append(dict(scenario=s, seed=seed, what="QT", choice=f"eta={eta},M={M}", eta=eta, M=M,
                                 set_cov=r["set_cov"], served=r["served"], overhead=r["overhead"]))
    for pol in GRIDS:
        r = S.simulate(Pv, pwr[va], RHO, sc, 0.1, pol, rng=np.random.default_rng(seed), seq=seq[va])
        rows.append(dict(scenario=s, seed=seed, what="grid", choice=f"{pol[0]}:{pol[1]}", served=r["served"],
                         overhead=r["overhead"]))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results/drift")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = Parallel(n_jobs=a.jobs)(delayed(job)(a.data, s, sd) for s in range(1, 10) for sd in range(a.seeds))
    df = pd.DataFrame([r for rows in res for r in rows]); df.to_csv(f"{a.out}/validation.csv", index=False)
    sc = df[df.what == "score"].groupby(["choice", "scenario"])[["set_cov", "size"]].mean()
    print("(1) static set on the validation stream, coverage and mean size per scenario:")
    print(sc.unstack(0).round(2).to_string())
    q = df[df.what == "QT"].groupby(["eta", "M", "scenario"])[["set_cov", "overhead"]].mean().groupby(["eta", "M"]).agg(["mean", "min"])
    q.columns = ["cov_mean", "cov_worst", "ovh_mean", "ovh_min"]
    q["score"] = (q.cov_mean - 0.9).abs() + 0.5 * q.ovh_mean           # rule fixed before looking: near 0.9, then cheap
    best = q.score.idxmin()
    print("\n(2) QT + certificates (best 5 by the rule):"); print(q.sort_values("score").head(5).round(3).to_string())
    print(f"    selected on validation: eta={best[0]}, M={best[1]}")
    # the same rules with one scenario left out: the choices for a cross-session stream whose target is that scenario
    rows = []
    for lo in [None, 1, 2, 3, 4, 8, 9]:
        d = df if lo is None else df[df.scenario != lo]
        qq = d[d.what == "QT"].groupby(["eta", "M", "scenario"])[["set_cov", "overhead"]].mean().groupby(["eta", "M"]).mean()
        e, m_ = ((qq.set_cov - 0.9).abs() + 0.5 * qq.overhead).idxmin()
        gg = d[d.what == "grid"].groupby(["choice", "scenario"])[["served", "overhead"]].mean()
        gg = gg.groupby("choice").agg(srv_worst=("served", "min"), ovh=("overhead", "mean"))
        pick = {}
        for fam in ["uniform:", "hybrid:"]:
            f = gg[gg.index.str.startswith(fam) & (gg.srv_worst >= 0.90)]
            pick[fam[:-1]] = f.ovh.idxmin() if len(f) else ""
        rows.append(dict(left_out=lo if lo else 0, eta=e, M=int(m_), grid=pick["uniform"], hybrid=pick["hybrid"]))
    C = pd.DataFrame(rows); C.to_csv(f"{a.out}/choices.csv", index=False)
    print("\n(4) choices with one scenario left out (0 = none):"); print(C.to_string(index=False))
    g = df[df.what == "grid"].groupby(["choice", "scenario"])[["served", "overhead"]].mean()
    g = g.groupby("choice").agg(srv_worst=("served", "min"), ovh=("overhead", "mean"))
    print("\n(3) cheapest policy whose worst validation stream has served coverage >= 0.90:")
    for fam in ["uniform:", "hybrid:"]:
        f = g[g.index.str.startswith(fam) & (g.srv_worst >= 0.90)]
        print(f"    {fam:9s}", f"{f.ovh.idxmin()} (validation: worst {f.loc[f.ovh.idxmin(), 'srv_worst']:.3f}, ovh {100*f.ovh.min():.1f}%)"
              if len(f) else "none")


if __name__ == "__main__":
    main()
