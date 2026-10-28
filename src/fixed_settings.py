# Operating points chosen on validation drives only, for both codebooks and several worst-stream targets.
# For each family (online set, sweeps-only IQT, fixed top-K, grid and hybrid from beam 0 [64 beams], grid and hybrid
# with a tuned start) and target, the member is the cheapest one whose worst VALIDATION stream (drives 55-80% of each
# scenario; train 0-40%, calibrate 40-55%) reaches the target. For a cross-session stream a->b the target scenario b
# is left out of that worst case. The chosen members are then read from the test results; the mass rule has no
# setting. This is the deployment-style counterpart of the test-selected frontier of the bootstrap scripts.
# xsel = "lto": a cross-session stream uses the validation drives without its target scenario;
# xsel = "lopo": a cross-session stream uses the test results of the cross-session streams of the other two
# base-station pairs (a session change seen at another site), never its own pair.
# Usage: python fixed_settings.py --data Position-Beam-Prediction/Gathered_data_DEV --results results --out results/fixed_settings
import argparse, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from settings import ETA, M_SWEEP
from streams import PAIRS
RHO = B.TARGETS["1dB"]; TARGETS = [0.8, 0.85, 0.9, 0.95]
STREAMS = [f"E{s}" for s in range(1, 10)] + [f"X{a}-{b}" for a, b in PAIRS]
ONLINE = [("cert", a) for a in [0.02, 0.05, 0.1, 0.2]] + [("sweep", a) for a in [0.02, 0.05, 0.1, 0.2]]
TOPK = {64: [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48], 16: [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]}


def job(D, s, seed, nb):
    B.NB = nb
    X, pwr, seq = B.load(D, s); pwr = pwr[:, ::64 // nb]; y = pwr.argmax(1); u = np.unique(seq); k = len(u)
    tr = np.where(np.isin(seq, u[:int(.4 * k)]))[0]; ca = np.where(np.isin(seq, u[int(.4 * k):int(.55 * k)]))[0]
    va = np.where(np.isin(seq, u[int(.55 * k):int(.8 * k)]))[0]
    Pc, Pv = B.fit_predict("mlp", X, y, tr, [ca, va], seed); sc = B.scores(Pc, B.good_mask(pwr[ca], RHO), "rank"); rows = []
    for fb, al in ONLINE:
        for ss in range(2):
            r = S.simulate(Pv, pwr[va], RHO, sc, al, ("QT", fb), M=M_SWEEP, eta=ETA, seq=seq[va],
                           rng=np.random.default_rng(1000 * seed + 10 * ss + s + nb))
            rows.append(dict(nbeams=nb, scenario=s, seed=seed, family="online" if fb == "cert" else "iqt", name=f"QT-{fb}:{al}",
                             served=r["served"], overhead=r["overhead"]))
    for K in TOPK[nb]:
        r = S.simulate(Pv, pwr[va], RHO, sc, 0.1, ("fixed", K), rng=np.random.default_rng(seed), seq=seq[va])
        rows.append(dict(nbeams=nb, scenario=s, seed=seed, family="topK", name=f"fixed:{K}", served=r["served"], overhead=r["overhead"]))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/fixed_settings"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=-1); ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); R = a.results
    vf = f"{a.out}/validation_online_topk.csv"
    if a.reuse and os.path.exists(vf): v = pd.read_csv(vf)
    else:
        res = Parallel(n_jobs=a.jobs)(delayed(job)(a.data, s, sd, nb) for nb in [64, 16] for s in range(1, 10) for sd in range(a.seeds))
        v = pd.DataFrame([r for rows in res for r in rows]); v.to_csv(vf, index=False)
    V = [v.groupby(["nbeams", "family", "name", "scenario"])[["served", "overhead"]].mean().reset_index()]
    g = pd.read_csv(f"{R}/drift/validation.csv"); g = g[g.what == "grid"]                     # grids from beam 0, 64 beams
    g = g.groupby(["choice", "scenario"])[["served", "overhead"]].mean().reset_index().rename(columns={"choice": "name"})
    g = g[g.name.str.startswith("uniform:") | g.name.str.startswith("hybrid:")]
    g["family"] = np.where(g.name.str.startswith("uniform"), "grid0", "hybrid0"); g["nbeams"] = 64; V.append(g)
    for part, nb in [("cb64", 64), ("cb16", 16)]:                                                # grids with a tuned start
        c = pd.read_csv(f"{R}/tuned_grids/choices_{part}.csv"); c = c[c.noise == 0].copy()
        c["name"] = "tuned-" + c.scheme + ":" + c.d.astype(str); c["family"] = np.where(c.scheme == "uniform", "tuned_grid", "tuned_hybrid")
        c["nbeams"] = nb; V.append(c[["nbeams", "family", "name", "scenario", "served", "overhead"]])
    V = pd.concat(V)
    T = {}                                                                                   # test results, per stream
    m64 = pd.read_csv(f"{R}/main/results_cb64.csv"); cn = pd.read_csv(f"{R}/coarse_noise/results_cb16.csv"); m16 = pd.read_csv(f"{R}/main/results_cb16.csv")
    t16 = pd.concat([cn, m16[~m16.name.isin(cn.name.unique())]])
    for nb, t, part, src in [(64, m64, "cb64", "main_runs"), (16, t16, "cb16", "coarse_noise_runs")]:
        t = t[(t.noise == 0) & (t.remeasure == 0)]
        lo = f"{R}/leave_target_out/results_leave_target_out.csv"
        if os.path.exists(lo):
            L = pd.read_csv(lo)
            if len(L):
                L = L[(L.source == src) & (L.part == part) & (L.noise == 0) & (L.remeasure == 0) & L.stream.isin(t.stream.unique())]
                t = pd.concat([t[~(t.stream.isin(L.stream.unique()) & t.name.str.startswith("QT"))], L])
        tg = pd.read_csv(f"{R}/tuned_grids/results_{part}.csv"); tg = tg[tg.noise == 0]
        T[nb] = pd.concat([t, tg]).groupby(["name", "stream"])[["served", "overhead"]].mean()
    out, per = [], []
    pair_of = lambda st: frozenset(int(v) for v in st[1:].split("-"))
    for nb in [64, 16]:
        for fam in ["online", "iqt", "topK", "grid0", "hybrid0", "tuned_grid", "tuned_hybrid"]:
            vv = vv0 = V[(V.nbeams == nb) & (V.family == fam)]
            vv = vv0[vv0.name.isin(T[nb].index.get_level_values(0))]
            if not len(vv): continue
            avail = [x for x in STREAMS if x in T[nb].index.get_level_values(1)]
            for xsel in ["lto", "lopo"]:
                for tau in TARGETS:
                    rows = []
                    for st in avail:
                        others = [x for x in avail if x[0] == "X" and pair_of(x) != pair_of(st)] if st[0] == "X" else []
                        if st[0] == "X" and xsel == "lopo" and others:
                            # new session: choose on the cross-session streams of the OTHER base-station pairs
                            tt = T[nb].reset_index(); tt = tt[tt.stream.isin(others) & tt.name.isin(vv.name.unique())]
                            w = tt.groupby("name").agg(worst=("served", "min"), ovh=("overhead", "mean"))
                        else:
                            lv = int(st.split("-")[1]) if st[0] == "X" else -1
                            w = vv[vv.scenario != lv].groupby("name").agg(worst=("served", "min"), ovh=("overhead", "mean"))
                        ok = w[w.worst >= tau]; ch = ok.ovh.idxmin() if len(ok) else w.worst.idxmax()
                        r = T[nb].loc[(ch, st)]; rows.append(dict(nbeams=nb, family=fam, xsel=xsel, target=tau, stream=st, choice=ch,
                                                                 val_ok=bool(len(ok)), served=r.served, overhead=r.overhead))
                    P = pd.DataFrame(rows); per += rows; E = P[P.stream.str[0] == "E"]; X_ = P[P.stream.str[0] == "X"]
                    out.append(dict(nbeams=nb, family=fam, xsel=xsel, target=tau, choices=";".join(sorted(P.choice.unique())),
                                    val_ok=P.val_ok.all(), worst=P.served.min(), worst_stream=P.loc[P.served.idxmin(), "stream"],
                                    ovh=100 * P.overhead.mean(), worst_E=E.served.min(), ovh_E=100 * E.overhead.mean(),
                                    choice_E=";".join(sorted(E.choice.unique())), worst_X=X_.served.min() if len(X_) else np.nan,
                                    ovh_X=100 * X_.overhead.mean() if len(X_) else np.nan, choice_X=";".join(sorted(X_.choice.unique())),
                                    met=P.served.min() >= tau))
    pd.DataFrame(per).to_csv(f"{a.out}/chosen_per_stream.csv", index=False)
    O = pd.DataFrame(out); O.to_csv(f"{a.out}/chosen.csv", index=False)
    pm = f"{R}/mass_rule/results_mass_rule.csv"
    if os.path.exists(pm):
        p = pd.read_csv(pm); p = p[p.protocol != "A"].groupby(["nbeams", "name", "stream"])[["served", "overhead"]].mean()
        print("mass rule (no setting):"); print(p.groupby(["nbeams", "name"]).agg(worst=("served", "min"), ovh=("overhead", "mean")).round(3))
    print(O.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
