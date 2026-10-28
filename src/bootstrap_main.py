# Paired block bootstrap over drives: uncertainty of worst-stream served coverage and of the overhead needed for
# worst-stream served coverage >= 0.9, per family. Resamples whole drives within each stream, the SAME drives for
# every scheme in a replicate. Uses the per-sample traces of main_runs.py (part cb64, noise 0).
# Usage: python bootstrap_main.py results/main [n_boot]
import pickle, sys
import numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "results/main"; NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
tr = pickle.load(open(f"{R}/traces_cb64.pkl", "rb"))
# aggregate per (stream, policy, drive): sums over all runs of served hits, served samples, probes, samples
acc = {}
for stream, seed, traces in tr:
    for name, runs in traces.items():
        for r in runs:
            a = acc.setdefault((stream, name), {})
            for dv in np.unique(r["drive_all"]):
                k = r["drive"] == dv; ka = r["drive_all"] == dv; s = a.setdefault(dv, np.zeros(4))
                s += [r["served"][k].sum(), k.sum(), r["probes"][ka].sum(), ka.sum()]
streams = sorted({s for s, _ in acc}); names = sorted({n for _, n in acc})
drives = {s: np.array(sorted(acc[(s, names[0])].keys())) for s in streams}
FAM = {"grid": ["uniform:5", "uniform:6", "uniform:8", "uniform:10"],
       "hybrid": [n for n in names if n.startswith("hybrid:")],
       "oracle": ["QT-oracle:0.02", "QT-oracle:0.05", "QT-oracle:0.1"], "ours": ["QT-cert:0.02", "QT-cert:0.05", "QT-cert:0.1"],
       "topK": [n for n in names if n.startswith("fixed:") and n != "fixed:64"]}
NBEAMS = 64


def stats(weights):
    """worst-stream served and mean overhead for every policy, given per-stream drive weights"""
    out = {}
    for n in names:
        srv, ovh = [], []
        for s in streams:
            M = np.array([acc[(s, n)][dv] for dv in drives[s]]); w = weights[s]
            srv.append((w * M[:, 0]).sum() / (w * M[:, 1]).sum()); ovh.append((w * M[:, 2]).sum() / (w * M[:, 3]).sum() / NBEAMS)
        out[n] = (min(srv), np.mean(ovh))
    return out


def needed(st, fam):
    ok = [st[n][1] for n in FAM[fam] if n in st and st[n][0] >= 0.90]
    return min(ok) if ok else np.inf


rng = np.random.default_rng(0)
point = stats({s: np.ones(len(drives[s])) for s in streams})
boots = []
for b in range(NBOOT):
    wts = {s: np.bincount(rng.integers(0, len(drives[s]), len(drives[s])), minlength=len(drives[s])).astype(float) for s in streams}
    boots.append(stats(wts))
rows = []
for n in names:
    w = np.array([bt[n][0] for bt in boots])
    rows.append(dict(policy=n, worst=point[n][0], lo=np.percentile(w, 2.5), hi=np.percentile(w, 97.5), ovh=100 * point[n][1],
                     p_ge_090=(w >= 0.9).mean()))
T = pd.DataFrame(rows).sort_values("ovh"); T.to_csv(f"{R}/bootstrap_policies.csv", index=False)
print("worst-stream served coverage, point and 95% interval over drive resamples:"); print(T.round(3).to_string(index=False))
need = {f: np.array([needed(bt, f) for bt in boots]) for f in FAM}
print("\noverhead needed for worst-stream served >= 0.90 (median, 95% interval, P[family never reaches 0.90]):")
for f, v in need.items():
    fin = v[np.isfinite(v)]
    print(f"  {f:7s} point {100*needed(point, f):5.1f}%  median {100*np.median(fin) if len(fin) else np.nan:5.1f}%  "
          f"[{100*np.percentile(fin, 2.5) if len(fin) else np.nan:5.1f}, {100*np.percentile(fin, 97.5) if len(fin) else np.nan:5.1f}]  "
          f"P(never)={np.mean(~np.isfinite(v)):.2f}")
for a, b in [("hybrid", "grid"), ("grid", "oracle"), ("hybrid", "oracle"), ("grid", "ours")]:
    print(f"  P[{a} needs less overhead than {b}] = {np.mean(need[a] < need[b]):.3f}")
pd.DataFrame({f: v for f, v in need.items()}).to_csv(f"{R}/bootstrap_needed.csv", index=False)
