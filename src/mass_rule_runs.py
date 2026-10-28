# Probability-mass baseline of prior work: for every sample, the smallest top set of beams whose predicted
# probabilities sum to tau = 0.9 (Morais et al., ICC 2023, with the raw model output; Hiam et al., with a
# temperature fitted on the calibration split). Reports set coverage (a beam within 1 dB, as in Table I, and the
# best beam, the reliability definition of Morais et al.) on the random split and on the two test protocols, and
# served coverage and overhead with the shared hill-climb on the 64- and 16-beam codebooks.
# Usage: python mass_rule_runs.py --data Position-Beam-Prediction/Gathered_data_DEV --out results/mass_rule
import argparse, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B
import beamsim as S
from streams import stream_data, PAIRS
RHO = B.TARGETS["1dB"]; TAU = 0.9


def kmass(P, tau, T):
    Z = np.clip(P, 1e-12, None) ** (1.0 / T); Z = Z / Z.sum(1, keepdims=True)
    return (np.cumsum(-np.sort(-Z, 1), 1) < tau).sum(1) + 1


def set_stats(P, pwr, tau, T):
    K = kmass(P, tau, T); Rk = B.ranks(P); G = B.good_mask(pwr, RHO); S_ = Rk <= K[:, None]
    return dict(set_cov=(S_ & G).any(1).mean(), best_cov=S_[np.arange(len(P)), pwr.argmax(1)].mean(), size=K.mean())


def job(D, stream, seed):
    rows, traces = [], {}
    for nb in [64, 16]:
        B.NB = nb
        X, pwr, seq, y, tr, ca, te = stream_data(D, stream, seed); pwr = pwr[:, ::64 // nb]; y = pwr.argmax(1)
        Pc, Pt = B.fit_predict("mlp", X, y, tr, [ca, te], seed, raw=True); T = B.fit_temperature(Pc, y[ca])
        for var, Tv in [("mass", 1.0), ("mass-ts", T)]:
            r, t_ = S.simulate(Pt, pwr[te], RHO, np.array([1.0]), 0.1, ("pmass", (TAU, Tv)), rng=np.random.default_rng(seed),
                               seq=seq[te], rec_traces=True)
            st = set_stats(Pt, pwr[te], TAU, Tv)
            rows.append(dict(stream=stream, seed=seed, nbeams=nb, protocol="E" if stream[0] == "E" else "X", name=var, T=Tv,
                             served=r["served"], overhead=r["overhead"], worst_roll100=r["worst_roll100"],
                             longest_miss_run=r["longest_miss_run"], **st))
            traces[(nb, var)] = dict(drive=seq[te][t_["keep"]], served=t_["served"], drive_all=seq[te], probes=t_["probes"])
        if stream[0] == "E" and nb == 64:                      # random 60/20/20 split of the same scenario (protocol of prior work)
            s = int(stream[1:]); Xa, pa, qa = B.load(D, s); ya = pa.argmax(1)
            trA, caA, teA = B.splits(qa, np.random.default_rng(seed))["A_random"]
            PcA, PtA = B.fit_predict("mlp", Xa, ya, trA, [caA, teA], seed, raw=True); TA = B.fit_temperature(PcA, ya[caA])
            for var, Tv in [("mass", 1.0), ("mass-ts", TA)]:
                rows.append(dict(stream=stream, seed=seed, nbeams=64, protocol="A", name=var, T=Tv, **set_stats(PtA, pa[teA], TAU, Tv)))
    return rows, (stream, seed, traces)


def boot_worst(tr, nb, n_boot=1000, seed=0):
    """drive-level bootstrap of worst-stream served coverage and mean overhead for one policy"""
    rng = np.random.default_rng(seed); acc = {}
    for stream, sd, T_ in tr:
        r = T_[nb]; a = acc.setdefault(stream, {})
        for dv in np.unique(r["drive_all"]):
            k = r["drive"] == dv; ka = r["drive_all"] == dv; v = a.setdefault(dv, np.zeros(4))
            v += [r["served"][k].sum(), k.sum(), r["probes"][ka].sum(), ka.sum()]
    Mx = {s: np.array(list(d.values())) for s, d in acc.items()}
    def stat(w): return min((w[s] * M[:, 0]).sum() / (w[s] * M[:, 1]).sum() for s, M in Mx.items())
    reps = [stat({s: np.bincount(rng.integers(0, len(M), len(M)), minlength=len(M)) for s, M in Mx.items()}) for _ in range(n_boot)]
    return np.percentile(reps, 2.5), np.percentile(reps, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results/mass_rule")
    ap.add_argument("--streams", nargs="*", default=[f"E{s}" for s in range(1, 10)] + [f"X{a}-{b}" for a, b in PAIRS])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--sweep-seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=-1); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(job)(a.data, st, sd) for st in a.streams for sd in range(a.seeds))
    df = pd.DataFrame([r for rows, _ in res for r in rows]); df.to_csv(f"{a.out}/results_mass_rule.csv", index=False)
    tr = [t for _, t in res]; ci = []
    for nb in [64, 16]:
        for var in ["mass", "mass-ts"]:
            lo, hi = boot_worst([(s, sd, {nb: T[(nb, var)]}) for s, sd, T in tr], nb, a.boot)
            ci.append(dict(nbeams=nb, name=var, lo=lo, hi=hi))
    pd.DataFrame(ci).to_csv(f"{a.out}/bootstrap_ci.csv", index=False)
    g = df.groupby(["nbeams", "protocol", "name", "stream"])[["set_cov", "best_cov", "size", "served", "overhead"]].mean()
    print(g.groupby(["nbeams", "protocol", "name"]).agg(["mean", "min"]).round(3).to_string()); print(pd.DataFrame(ci).round(3))


if __name__ == "__main__":
    main()
