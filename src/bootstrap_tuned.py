# Paired drive-level bootstrap for the tuned-start grids and hybrids (tuned_grid_runs.py), paired with the traces
# of main_runs.py (64 beams) and coarse_noise_runs.py (16 beams, probe noise). Same drives for every scheme in a
# replicate. "Overhead needed" for a target = mean overhead of the cheapest tested member of a family whose
# worst-stream served coverage reaches the target in that replicate (+inf = never).
# Usage: python bootstrap_tuned.py results/main results/coarse_noise results/tuned_grids [n_boot]
import pickle, sys
import numpy as np, pandas as pd
R_MAIN, R_CN, R_TG = sys.argv[1:4]; NBOOT = int(sys.argv[4]) if len(sys.argv) > 4 else 1000


def add(acc, path, keyf):
    for stream, seed, traces in pickle.load(open(path, "rb")):
        for key, runs in traces.items():
            k2 = keyf(key)
            if k2 is None: continue
            for r in runs:
                a = acc.setdefault((stream, k2), {})
                for dv in np.unique(r["drive_all"]):
                    k = r["drive"] == dv; ka = r["drive_all"] == dv; s = a.setdefault(dv, np.zeros(4))
                    s += [r["served"][k].sum(), k.sum(), r["probes"][ka].sum(), ka.sum()]


def boot(acc, nb, rng):
    keys = sorted({k for _, k in acc}); streams = sorted({s for s, _ in acc})
    drives = {s: np.array(sorted(acc[(s, keys[0])].keys())) for s in streams}
    M = {(s, k): np.array([acc[(s, k)][d] for d in drives[s]]) for s in streams for k in keys}
    def stats(w):
        out = {}
        for k in keys:
            srv = [(w[s] * M[(s, k)][:, 0]).sum() / (w[s] * M[(s, k)][:, 1]).sum() for s in streams]
            ovh = [(w[s] * M[(s, k)][:, 2]).sum() / (w[s] * M[(s, k)][:, 3]).sum() / nb for s in streams]
            out[k] = (min(srv), float(np.mean(ovh)))
        return out
    pt = stats({s: np.ones(len(drives[s])) for s in streams})
    reps = [stats({s: np.bincount(rng.integers(0, len(drives[s]), len(drives[s])), minlength=len(drives[s])).astype(float) for s in streams})
            for _ in range(NBOOT)]
    return keys, pt, reps


def needed(st, members, t):
    ok = [st[k][1] for k in members if st[k][0] >= t]
    return min(ok) if ok else np.inf


def summ(v):
    f = v[np.isfinite(v)]
    return (100 * np.median(f) if len(f) else np.nan, 100 * np.percentile(f, 2.5) if len(f) else np.nan,
            100 * np.percentile(f, 97.5) if len(f) else np.nan, np.mean(~np.isfinite(v)))


rng = np.random.default_rng(0); rows_ci, rows_need, rows_p = [], [], []
def record(part, noise, keys, pt, reps, fams, targets, pairs):
    for k in keys:
        w = np.array([r[k][0] for r in reps])
        rows_ci.append(dict(part=part, noise=noise, name=k, worst=pt[k][0], lo=np.percentile(w, 2.5), hi=np.percentile(w, 97.5), ovh=100 * pt[k][1]))
    nd = {}
    for t in targets:
        for f, mem in fams.items():
            mem = [m for m in mem if m in keys]
            if not mem: continue
            v = np.array([needed(r, mem, t) for r in reps]); nd[(t, f)] = v; md, lo, hi, nev = summ(v)
            rows_need.append(dict(part=part, noise=noise, target=t, family=f, point=100 * needed(pt, mem, t), median=md, lo=lo, hi=hi, p_never=nev))
        for a, b in pairs:
            bs = b if isinstance(b, tuple) else (b,)
            if (t, a) in nd and all((t, x) in nd for x in bs):
                bb = np.minimum.reduce([nd[(t, x)] for x in b]) if isinstance(b, tuple) else nd[(t, b)]
                rows_p.append(dict(part=part, noise=noise, target=t, a=a, b=str(b), p_a_cheaper=np.mean(nd[(t, a)] < bb),
                                   p_tie=np.mean((nd[(t, a)] == bb) & np.isfinite(bb))))


# ---- 64 beams, noise-free: tuned grids/hybrids vs the main-run schemes ----
acc = {}
add(acc, f"{R_MAIN}/traces_cb64.pkl", lambda k: k if isinstance(k, str) else None)
add(acc, f"{R_TG}/traces_cb64.pkl", lambda k: k[1] if k[0] == 0.0 else None)
keys, pt, reps = boot(acc, 64, rng)
fam = {"tuned_grid": [k for k in keys if k.startswith("tuned-uniform:")], "tuned_hybrid": [k for k in keys if k.startswith("tuned-hybrid")],
       "grid0": [k for k in keys if k.startswith("uniform:")], "hybrid0": [k for k in keys if k.startswith("hybrid:")],
       "oracle": [k for k in keys if k.startswith("QT-oracle:")], "ours": [k for k in keys if k.startswith("QT-cert:")],
       "topK": [k for k in keys if k.startswith("fixed:") and k != "fixed:64"]}
record("cb64", 0.0, keys, pt, reps, fam, [0.9, 0.93, 0.95], [("tuned_hybrid", "tuned_grid"), ("tuned_grid", "tuned_hybrid"),
                                                            ("tuned_grid", "oracle"), ("tuned_hybrid", "oracle")])
del acc
# ---- 16 beams, noise-free: grids/hybrids = start at beam 0 OR tuned (whichever is cheaper) ----
acc = {}
add(acc, f"{R_CN}/traces_cb16.pkl", lambda k: k[1] if k[0] == 0.0 else None)
add(acc, f"{R_TG}/traces_cb16.pkl", lambda k: k[1] if k[0] == 0.0 else None)
keys, pt, reps = boot(acc, 16, rng)
fam = {"ours": [k for k in keys if k.startswith("QT-cert:")], "oracle": [k for k in keys if k.startswith("QT-oracle:")],
       "iqt": [k for k in keys if k.startswith("QT-sweep:")], "topK": [k for k in keys if k.startswith("fixed:") and k != "fixed:16"],
       "grid": [k for k in keys if k.startswith("uniform:") or k.startswith("tuned-uniform:")],
       "hybrid": [k for k in keys if k.startswith("hybrid:") or k.startswith("tuned-hybrid")],
       "tuned_grid": [k for k in keys if k.startswith("tuned-uniform:")], "tuned_hybrid": [k for k in keys if k.startswith("tuned-hybrid")]}
record("cb16", 0.0, keys, pt, reps, fam, [0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92],
       [("ours", ("topK", "grid", "hybrid", "iqt")), ("oracle", ("topK", "grid", "hybrid", "iqt"))])
del acc
# ---- 64 beams under probe noise (re-measured) ----
for nz in [0.25, 0.5, 1.0]:
    acc = {}
    add(acc, f"{R_CN}/traces_cb64n.pkl", lambda k: k[1] if k[0] == nz else None)
    add(acc, f"{R_TG}/traces_cb64n.pkl", lambda k: k[1] if k[0] == nz else None)
    keys, pt, reps = boot(acc, 64, rng)
    fam = {"grid": [k for k in keys if k.startswith("uniform:") or k.startswith("tuned-uniform:")],
           "hybrid": [k for k in keys if k.startswith("hybrid:") or k.startswith("tuned-hybrid")],
           "ours": [k for k in keys if k.startswith("QT-cert:")], "iqt": [k for k in keys if k.startswith("QT-sweep:")],
           "topK": ["fixed:32"], "exhaustive": ["fixed:64"]}
    record("cb64n", nz, keys, pt, reps, fam, [0.85, 0.9], [("ours", ("grid", "hybrid", "topK", "iqt"))])
    del acc
pd.DataFrame(rows_ci).to_csv(f"{R_TG}/bootstrap_ci.csv", index=False)
pd.DataFrame(rows_need).to_csv(f"{R_TG}/bootstrap_needed.csv", index=False)
pd.DataFrame(rows_p).to_csv(f"{R_TG}/bootstrap_pairs.csv", index=False)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows_need).round(3).to_string(index=False)); print(pd.DataFrame(rows_p).round(3).to_string(index=False))
c = pd.DataFrame(rows_ci); print(c[c.name.str.startswith("tuned")].round(3).to_string(index=False))
