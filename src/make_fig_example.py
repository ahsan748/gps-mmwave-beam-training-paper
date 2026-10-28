# Fig. 1: three real 64-beam power profiles (scenario 9, future-drive protocol, seed 0, static split-conformal set):
# (a) the set covers, (b) a certified miss (the hill-climb finds a beam >= 1 dB stronger),
# (c) an uncertified miss (a second lobe; only a sweep reveals it).
# Usage: python make_fig_example.py --data Position-Beam-Prediction/Gathered_data_DEV --out outputs
import argparse, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamlib as B

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results/figs")
ap.add_argument("--scen", type=int, default=9); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
X, pwr, seq = B.load(a.data, a.scen); y = pwr.argmax(1)
tr, ca, te = B.splits(seq, np.random.default_rng(0))["E_blocked_future"]
Pc, Pt = B.fit_predict("mlp", X, y, tr, [ca, te], 0); rho = B.TARGETS["1dB"]
Gc, Gt = B.good_mask(pwr[ca], rho), B.good_mask(pwr[te], rho)
q = B.qhat(B.scores(Pc, Gc, "rank"), 0.1); R = B.ranks(Pt); P = pwr[te]


def climb_path(p, S):
    probes = set(np.flatnonzero(S)); b = max(probes, key=lambda j: p[j]); path = [b]; extra = []
    while True:
        nb = [j for j in (b - 1, b + 1) if 0 <= j < 64]; new = [j for j in nb if j not in probes]
        extra += new; probes.update(new); j = max(nb, key=lambda j: p[j])
        if p[j] > p[b]: b = j; path.append(b)
        else: return path, extra


cases = {"covered": [], "cert": [], "uncert": []}
for i in range(len(P)):
    S = R[i] <= q; best = P[i][S].max(); hit = (S & Gt[i]).any()
    path, _ = climb_path(P[i], S); cert = P[i][path[-1]] >= B.CERT * best
    key = "covered" if hit else ("cert" if cert else "uncert")
    # prefer clear examples: a single sharp lobe for (a)/(b), a well-separated second lobe for (c)
    gap = abs(int(P[i].argmax()) - int(np.flatnonzero(S).mean()))
    if key == "uncert":                                     # a separate lobe: a deep dip between the set and the best beam
        db = 10 * np.log10(P[i] / P[i].max()); a_, b_ = sorted([int(P[i].argmax()), int(np.flatnonzero(S)[np.argmax(P[i][S])])])
        depth = 10 * np.log10(best / P[i].max()) - db[a_:b_ + 1].min()
        gap = gap * (depth >= 4) + depth / 100
    cases[key].append((len(path) if key == "cert" else gap, i))
pick = {"covered": min(cases["covered"])[1], "cert": max(cases["cert"])[1],
        "uncert": max(cases["uncert"])[1]}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(3.5, 1.6), sharey=True)
titles = {"covered": "(a) set covers", "cert": "(b) certified miss", "uncert": "(c) uncertified miss"}
for a_, key in zip(ax, ["covered", "cert", "uncert"]):
    i = pick[key]; p = P[i]; S = R[i] <= q; db = 10 * np.log10(p / p.max())
    path, extra = climb_path(p, S)
    a_.plot(np.arange(64), db, color="#8c8c8c", lw=0.8)
    a_.axhline(-1, color="k", lw=0.5, ls="--")
    a_.scatter(np.flatnonzero(S), db[S], s=12, color="#2a78d6", zorder=3, label="predicted set")
    if extra: a_.scatter(extra, db[extra], s=12, marker="^", color="#eb6834", zorder=3, label="hill-climb probes")
    a_.scatter([p.argmax()], [0], s=22, marker="*", color="k", zorder=4, label="best beam")
    a_.set_title(titles[key], fontsize=6.5); a_.set_xlabel("Beam index"); a_.set_xticks([0, 32, 63])
    a_.set_ylim(-11, 0.8)
ax[0].set_ylabel("Power rel. best (dB)")
h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[1].get_legend_handles_labels()
lab = dict(zip(l + l2, h + h2))
fig.legend(lab.values(), lab.keys(), ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6)
fig.tight_layout(pad=0.3, w_pad=0.3)
fig.savefig(f"{a.out}/fig0_examples.pdf", bbox_inches="tight"); fig.savefig(f"{a.out}/fig0_examples.png", dpi=220, bbox_inches="tight")
print("set size q =", q, "| picked test samples:", pick, "| counts:", {k: len(v) for k, v in cases.items()})
