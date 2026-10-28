# Paired drive-level bootstrap for the coarse-codebook and probe-noise runs (coarse_noise_runs.py).
# For each replicate, whole drives are resampled within each stream, the SAME drives for every scheme.
# "Overhead needed" = mean overhead of the cheapest TESTED member of a family whose worst-stream served coverage
# is >= the target in that replicate; a replicate in which no member reaches the target counts as +inf (never).
# Usage: python bootstrap_coarse_noise.py results/coarse_noise [n_boot]
import pickle, sys
import numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "results/coarse_noise"; NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 1000


def load(part):
    acc = {}
    for stream, seed, traces in pickle.load(open(f"{R}/traces_{part}.pkl", "rb")):
        for key, runs in traces.items():
            for r in runs:
                a = acc.setdefault((stream, key), {})
                for dv in np.unique(r["drive_all"]):
                    k = r["drive"] == dv; ka = r["drive_all"] == dv; s = a.setdefault(dv, np.zeros(4))
                    s += [r["served"][k].sum(), k.sum(), r["probes"][ka].sum(), ka.sum()]
    return acc


def boot(acc, nb, keys, rng):
    streams = sorted({s for s, _ in acc}); drives = {s: np.array(sorted(acc[(s, keys[0])].keys())) for s in streams}
    def stats(w):
        out = {}
        for k in keys:
            srv, ovh = [], []
            for s in streams:
                M = np.array([acc[(s, k)][d] for d in drives[s]]); ww = w[s]
                srv.append((ww * M[:, 0]).sum() / (ww * M[:, 1]).sum()); ovh.append((ww * M[:, 2]).sum() / (ww * M[:, 3]).sum() / nb)
            out[k] = (min(srv), float(np.mean(ovh)))
        return out
    point = stats({s: np.ones(len(drives[s])) for s in streams}); reps = []
    for _ in range(NBOOT):
        reps.append(stats({s: np.bincount(rng.integers(0, len(drives[s]), len(drives[s])), minlength=len(drives[s])).astype(float)
                           for s in streams}))
    return point, reps


def needed(st, members, target):
    ok = [st[k][1] for k in members if k in st and st[k][0] >= target]
    return min(ok) if ok else np.inf


rng = np.random.default_rng(0); out_rows = []
# ---- 16-beam codebook, noise-free ----
acc = load("cb16"); keys16 = sorted({k for _, k in acc if k[0] == 0.0})
pt, reps = boot({kk: v for kk, v in acc.items() if kk[1][0] == 0.0}, 16, keys16, rng)
for k in keys16:
    w = np.array([r[k][0] for r in reps])
    out_rows.append(dict(part="cb16", noise=0.0, name=k[1], worst=pt[k][0], lo=np.percentile(w, 2.5), hi=np.percentile(w, 97.5), ovh=100 * pt[k][1]))
fam16 = {"topK": [k for k in keys16 if k[1].startswith("fixed:") and k[1] != "fixed:16"],
         "grid": [k for k in keys16 if k[1].startswith("uniform:")], "hybrid": [k for k in keys16 if k[1].startswith("hybrid:")],
         "ours": [k for k in keys16 if k[1].startswith("QT-cert:")], "oracle": [k for k in keys16 if k[1].startswith("QT-oracle:")],
         "iqt": [k for k in keys16 if k[1].startswith("QT-sweep:")]}
print("16-beam codebook: overhead needed for worst-stream served >= target (median [95% CI], P(never))")
need16 = {}
TGTS = [0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92]
for tgt in TGTS:
    for f, mem in fam16.items():
        v = np.array([needed(r, mem, tgt) for r in reps]); need16[(tgt, f)] = v; fin = v[np.isfinite(v)]
        print(f"  target {tgt:.2f} {f:6s} point {100*needed(pt, mem, tgt):5.1f}%  median {100*np.median(fin) if len(fin) else np.nan:5.1f}%  "
              f"[{100*np.percentile(fin,2.5) if len(fin) else np.nan:5.1f}, {100*np.percentile(fin,97.5) if len(fin) else np.nan:5.1f}]  P(never)={np.mean(~np.isfinite(v)):.2f}")
curve = []
for tgt in TGTS:
    others = np.minimum.reduce([need16[(tgt, f)] for f in ["topK", "grid", "hybrid", "iqt"]])
    curve.append(dict(target=tgt, p_ours_cheapest=np.mean(need16[(tgt, "ours")] < others), p_ours_never=np.mean(~np.isfinite(need16[(tgt, "ours")])),
                      p_oracle_cheapest=np.mean(need16[(tgt, "oracle")] < others)))
pd.DataFrame(curve).to_csv(f"{R}/bootstrap16_curve.csv", index=False); print(pd.DataFrame(curve).round(3).to_string(index=False))
for tgt in [0.85, 0.88, 0.90]:
    for a, b in [("ours", "topK"), ("oracle", "topK"), ("ours", "grid"), ("ours", "hybrid"), ("ours", "iqt")]:
        print(f"  target {tgt:.2f}: P[{a} needs less overhead than {b}] = {np.mean(need16[(tgt, a)] < need16[(tgt, b)]):.3f}")
pd.DataFrame({f"{t}_{f}": v for (t, f), v in need16.items()}).to_csv(f"{R}/bootstrap16_needed.csv", index=False)
# ---- 64-beam under probe noise (re-measured): intervals of worst-stream served ----
acc = load("cb64n")
for nz in [0.25, 0.5, 1.0]:
    ks = sorted({k for _, k in acc if k[0] == nz})
    pt, reps = boot({kk: v for kk, v in acc.items() if kk[1][0] == nz}, 64, ks, rng)
    for k in ks:
        w = np.array([r[k][0] for r in reps])
        out_rows.append(dict(part="cb64n", noise=nz, name=k[1], worst=pt[k][0], lo=np.percentile(w, 2.5), hi=np.percentile(w, 97.5), ovh=100 * pt[k][1]))
T = pd.DataFrame(out_rows); T.to_csv(f"{R}/bootstrap_ci.csv", index=False)
print("\nworst-stream served coverage with 95% intervals:"); print(T.round(3).to_string(index=False))
