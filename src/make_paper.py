# Builds the six tables and Fig. 2 of the paper from the result files, plus numbers.txt with the figures quoted
# in the text. Run after all experiment scripts.  Usage: python make_paper.py results outputs
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker

R, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
rd = lambda p: pd.read_csv(f"{R}/{p}")
NUM = []
def num(key, val): NUM.append(f"{key}: {val}")


def per_stream(df, cols=("served", "overhead")):
    return df.groupby(["name", "stream"])[list(cols)].mean().reset_index()


def summary(df):
    """per scheme: mean and worst stream of the seed-averaged coverage, mean overhead; bursts over every single run"""
    g = df.groupby(["name", "stream"])[["served", "overhead", "set_cov", "ovh_p95"]].mean()
    out = g.groupby("name").agg(mean=("served", "mean"), worst=("served", "min"), ovh=("overhead", "mean"),
                                set_mean=("set_cov", "mean"), set_worst=("set_cov", "min"), p95=("ovh_p95", "mean"))
    b = df.groupby("name").agg(run=("longest_miss_run", "max"), roll=("worst_roll100", "min"))
    return out.join(b)


def write(name, rows, head):
    with open(f"{OUT}/{name}.tex", "w") as fh:
        fh.write(head + "\\midrule\n" + "".join(r + "\\\\\n" for r in rows) + "\\bottomrule\n")


f2 = lambda v: f"{v:.2f}"; f3 = lambda v: f"{v:.3f}"; f1 = lambda v: f"{v:.1f}"
c64 = rd("main/results_cb64.csv"); z64 = c64[c64.noise == 0]; A64 = summary(z64)
c16 = rd("coarse_noise/results_cb16.csv"); z16 = c16[c16.noise == 0]; A16 = summary(z16)
cn = rd("coarse_noise/results_cb64n.csv")
tg = {p: rd(f"tuned_grids/results_{p}.csv") for p in ["cb64", "cb16", "cb64n"]}
T64 = summary(tg["cb64"][tg["cb64"].noise == 0]); T16 = summary(tg["cb16"][tg["cb16"].noise == 0])
ci_main = rd("main/bootstrap_policies.csv").set_index("policy")
ci_cn = rd("coarse_noise/bootstrap_ci.csv"); ci_tg = rd("tuned_grids/bootstrap_ci.csv")
def ci(part, noise, name):
    """95% interval of the worst-stream served coverage from the bootstrap that produced the table entry"""
    if part == "cb64" and not name.startswith("tuned-") and name in ci_main.index:
        x = ci_main.loc[name]; return x.lo, x.hi
    for d in ((ci_tg,) if name.startswith("tuned-") else (ci_cn, ci_tg)):
        x = d[(d.part == part) & (d.noise == noise) & (d.name == name)]
        if len(x): return x.lo.iloc[0], x.hi.iloc[0]
    raise KeyError(name)

# ---------- Sec. II: codebook ----------
ov = rd("coarse_noise/overlap.csv").set_index("nbeams")
for nb in [64, 16]: num(f"{nb} beams: best neighbour below best (dB) / beams within 1 dB", f"{-ov.loc[nb, 'neighbour_db']:.2f} / {ov.loc[nb, 'within1db']:.1f}")

# ---------- Table I: drift ----------
f = rd("drift/results_drift.csv"); st = f[(f.part == "drift") & (f.policy == "static")]
pm = rd("mass_rule/results_mass_rule.csv")
dr = st[st.predictor == "mlp"].groupby(["protocol", "stream"])[["set_cov", "size"]].mean()
dk = st[st.predictor == "knn"].groupby(["protocol", "stream"]).set_cov.mean()
pg = pm[pm.nbeams == 64].groupby(["name", "protocol", "stream"])[["set_cov", "best_cov", "size", "served", "overhead"]].mean()
rows = []
for p, lab in [("A", "Random split (as in~\\cite{morais2022position})"), ("E", "Past / future drives"), ("X", "One / other session")]:
    a, m = dr.loc[p], pg.loc[("mass", p)]
    rows.append(f"{lab} & {f2(a.set_cov.mean())} & {f2(a.set_cov.min())} & {f1(a['size'].mean())} & "
                f"{f2(m.set_cov.mean())} & {f2(m.set_cov.min())} & {f1(m['size'].mean())}")
write("table1_drift", rows, "Calibrated on / tested on & mean & worst & size & mean & worst & size\\\\\n")
num("kNN conformal worst set coverage, future / new session", f"{dk.loc['E'].min():.2f} / {dk.loc['X'].min():.2f}")
num("mass rule, random split: size / mean best-beam coverage", f"{pg.loc[('mass', 'A')]['size'].mean():.1f} / {pg.loc[('mass', 'A')].best_cov.mean():.2f}")
for nm in ["mass", "mass-ts"]:
    num(f"{nm}: worst set coverage future / new session", f"{pg.loc[(nm, 'E')].set_cov.min():.2f} / {pg.loc[(nm, 'X')].set_cov.min():.2f}")
num("mass: worst best-beam coverage future / new session", f"{pg.loc[('mass', 'E')].best_cov.min():.2f} / {pg.loc[('mass', 'X')].best_cov.min():.2f}")
t1 = f[f.predictor == "mlp"].groupby("stream").top1.mean(); num("top-1 accuracy range (%)", f"{100*t1.min():.0f}-{100*t1.max():.0f}")

# ---------- Table II: feedback ----------
rows = []
for lab, nm in [("No feedback (static)", "static:None"), ("Certificates only", "QT-certonly:0.1"),
                ("Sweeps only, IQT~\\cite{zhao2025iqt}", "QT-sweep:0.1"), ("ACI, same $e_t$", "ACI-cert:0.1"),
                ("Certificates + sweeps", "QT-cert:0.1"), ("Oracle (every miss known)$^\\ast$", "QT-oracle:0.1")]:
    z = z64[z64.name == nm]; g = z.groupby("stream")
    sp = (g.set_cov.max() - g.set_cov.min()).max() if z.sweep_seed.nunique() > 1 else np.nan
    a = A64.loc[nm]
    rows.append(f"{lab} & {f3(a.set_mean)} & {f3(a.set_worst)} & {'--' if np.isnan(sp) else f2(sp)} & {f1(100*a.ovh)} & {100*a.p95:.0f}")
write("table2_feedback", rows, "& \\multicolumn{2}{c}{Set coverage} & Run & \\multicolumn{2}{c}{Ovh.\\ (\\%)}\\\\\nFeedback & mean & worst & spread & mean & p95\\\\\n")
q = z64[z64.name == "QT-cert:0.1"]; qs = q.groupby("stream")
w = per_stream(z64[z64.name.isin(["QT-cert:0.1", "QT-sweep:0.1", "QT-oracle:0.1"])], ("set_cov", "overhead", "n_sweeps"))
ws = w[w.name == "QT-cert:0.1"].set_index("stream").set_cov.idxmin(); x = w[w.stream == ws].set_index("name")
num(f"worst stream of the online set ({ws}, {int(q[q.stream == ws].n_test.iloc[0])} samples, {x.loc['QT-cert:0.1', 'n_sweeps']:.1f} sweeps): set cov / ovh online, IQT, oracle",
    "; ".join(f"{x.loc[n, 'set_cov']:.3f} / {100*x.loc[n, 'overhead']:.1f}%" for n in ["QT-cert:0.1", "QT-sweep:0.1", "QT-oracle:0.1"]))
num("ACI: share of samples with all beams probed (%)", f"{100*z64[z64.name == 'ACI-cert:0.1'].frac_full.mean():.0f}")
num("|Delta_T| median / max; fewest sweeps on a stream", f"{q.delta_T.abs().median():.3f} / {q.delta_T.abs().max():.2f}; {qs.n_sweeps.mean().min():.0f}")
num("share of misses certified, range over streams", f"{qs.cert_recall.mean().min():.2f}-{qs.cert_recall.mean().max():.2f}")
for nm in ["QT-cert:0.1", "uniform:6"]: num(f"extra climb probes per sample, {nm}", f"{z64[z64.name == nm].climb_probes.mean():.1f}")
e = rd("eta/summary.csv")
for _, r in e.iterrows():
    num(f"eta {r.eta} {r.feedback} alpha {r.alpha}: set cov mean/worst, served worst, ovh",
        f"{r.set_cov_mean:.3f}/{r.set_cov_worst:.3f}, {r.served_worst:.3f}, {r.overhead:.1f}%")

# ---------- Table III: 64 beams ----------
go = rd("grid_start/results_offsets.csv").groupby(["scheme", "d", "offset", "stream"])[["served", "overhead"]].mean().reset_index()
gw = go.groupby(["scheme", "d", "offset"]).agg(worst=("served", "min"), ovh=("overhead", "mean")).reset_index()
gs = gw.groupby(["scheme", "d"]).agg(wmin=("worst", "min"), wmax=("worst", "max"), ovh=("ovh", "mean"))
rows = []
for lab, nm, start in [("Grid, every 6th", "uniform:6", ("uniform", 6)), ("Grid, every 8th", "uniform:8", ("uniform", 8)),
                       ("Grid, every 10th", "uniform:10", ("uniform", 10)), ("Top-1 + every 8th", "hybrid:(1, 8)", ("hybrid", 8)),
                       ("Top-1 + every 10th", "hybrid:(1, 10)", ("hybrid", 10)), ("Top-2 + every 16th", "hybrid:(2, 16)", ("hybrid2", 16)),
                       ("GPS top-32", "fixed:32", None), ("Mass rule, $\\tau{=}0.9$", None, None),
                       ("Online set, $\\alpha{=}0.02$", "QT-cert:0.02", None), ("Online set, $\\alpha{=}0.05$", "QT-cert:0.05", None),
                       ("Online set, $\\alpha{=}0.1$", "QT-cert:0.1", None), ("Oracle set, $\\alpha{=}0.05$", "QT-oracle:0.05", None),
                       ("Tracking (no GPS)", "track:None", None)]:
    if nm is None:
        p = pm[(pm.nbeams == 64) & (pm.name == "mass") & (pm.protocol != "A")].groupby("stream")[["served", "overhead"]].mean()
        b = rd("mass_rule/bootstrap_ci.csv").set_index(["nbeams", "name"]).loc[(64, "mass")]
        rows.append(f"{lab} & {f3(p.served.mean())} & {f3(p.served.min())} [{f2(b.lo)}, {f2(b.hi)}] & -- & -- & {f1(100*p.overhead.mean())}"); continue
    a = A64.loc[nm]
    cell = f3(a.worst) if nm == "track:None" else f"{f3(a.worst)} [{f2(ci('cb64', 0.0, nm)[0])}, {f2(ci('cb64', 0.0, nm)[1])}]"
    if start:
        tn = f"tuned-{start[0]}:{start[1]}"
        rows.append(f"{lab} & {f3(a['mean'])} & {cell} & {f3(gs.loc[start].wmin)} & {f3(T64.loc[tn].worst)} & {f1(100*a.ovh)}")
    else:
        rows.append(f"{lab} & {f3(a['mean'])} & {cell} & -- & -- & {f1(100*a.ovh)}")
write("table3_ops64", rows, "& & \\multicolumn{3}{c}{Worst-stream served coverage} & Ovh.\\\\\nScheme (64 beams) & mean & start 0 [95\\% CI] & worst st. & tuned$^\\dagger$ & (\\%)\\\\\n")
num("IQT alpha 0.05 / 0.02, 64 beams: worst / ovh", f"{A64.loc['QT-sweep:0.05'].worst:.3f} / {100*A64.loc['QT-sweep:0.05'].ovh:.1f}%; {A64.loc['QT-sweep:0.02'].worst:.3f} / {100*A64.loc['QT-sweep:0.02'].ovh:.1f}%")
mt = rd("mass_rule/results_mass_rule.csv"); mt = mt[(mt.protocol != "A") & (mt.name == "mass-ts")].groupby(["nbeams", "stream"])[["served", "overhead"]].mean()
for nb in [64, 16]: num(f"temperature-scaled mass rule, {nb} beams: worst / ovh", f"{mt.loc[nb].served.min():.3f} / {100*mt.loc[nb].overhead.mean():.1f}%")
kn = f[(f.predictor == "knn") & (f.part == "frontier") & (f.noise == 0) & (f.policy == "hybrid") & (f.par == "(1, 10)")]
kn = kn.groupby("stream")[["served", "overhead"]].mean(); num("kNN top-1 + every 10th: worst / ovh", f"{kn.served.min():.3f} / {100*kn.overhead.mean():.1f}%")
E = per_stream(z64); E = E[E.stream.str.startswith("E")].groupby("name").agg(w=("served", "min"), o=("overhead", "mean"))
for nm in ["QT-cert:0.05", "QT-oracle:0.05", "hybrid:(1, 10)", "uniform:6"]: num(f"future drives only, {nm}: worst / ovh", f"{E.loc[nm].w:.3f} / {100*E.loc[nm].o:.1f}%")
tu = rd("grid_start_tuned/tuned_offsets.csv").set_index(["scheme", "d"])
for k in [("uniform", 10), ("uniform", 8)]: num(f"future drives only, tuned {k}: worst / ovh", f"{tu.loc[k].worst_tuned_E:.3f} / {tu.loc[k].ovh_tuned_E:.1f}%")
for k in [("hybrid", 8), ("uniform", 8)]: num(f"all starts {k}: worst range / mean ovh", f"{gs.loc[k].wmin:.3f}-{gs.loc[k].wmax:.3f} / {100*gs.loc[k].ovh:.1f}%")
num("tuned start within 0.01 of best test start (streams of 15)", f"{15*tu.share_within_001.min():.0f}-{15*tu.share_within_001.max():.0f}")
for nm in ["tuned-uniform:10", "tuned-hybrid:10", "tuned-uniform:8"]:
    lo, hi = ci("cb64", 0.0, nm); num(f"{nm}: worst [CI] / ovh", f"{T64.loc[nm].worst:.3f} [{lo:.2f}, {hi:.2f}] / {100*T64.loc[nm].ovh:.1f}%")
pr = rd("tuned_grids/bootstrap_pairs.csv"); pr = pr[(pr.part == "cb64") & (pr.noise == 0)]
for t, a_, b_ in [(0.9, "tuned_hybrid", "tuned_grid"), (0.93, "tuned_grid", "tuned_hybrid")]:
    num(f"P({a_} cheaper than {b_}) at {t}", f"{100*pr[(pr.target == t) & (pr.a == a_) & (pr.b == b_)].p_a_cheaper.iloc[0]:.0f}%")
for nm in ["uniform:6", "QT-cert:0.05"]: num(f"bursts {nm}: worst 100-sample window / longest miss run", f"{A64.loc[nm].roll:.2f} / {A64.loc[nm].run:.0f}")

# ---------- Table IV: 16 beams ----------
rows = []
for lab, nm, src in [("GPS top-10", "fixed:10", A16), ("GPS top-12", "fixed:12", A16), ("Grid, every 2nd$^\\dagger$", "tuned-uniform:2", T16),
                     ("Grid, every 3rd$^\\dagger$", "tuned-uniform:3", T16), ("Top-1 + every 3rd$^\\dagger$", "tuned-hybrid:3", T16),
                     ("Sweeps only (IQT), $\\alpha{=}0.1$", "QT-sweep:0.1", A16), ("Online set, $\\alpha{=}0.02$", "QT-cert:0.02", A16),
                     ("Online set, $\\alpha{=}0.05$", "QT-cert:0.05", A16), ("Online set, $\\alpha{=}0.1$", "QT-cert:0.1", A16),
                     ("Oracle set, $\\alpha{=}0.05$", "QT-oracle:0.05", A16), ("Mass rule, $\\tau{=}0.9$", None, None)]:
    if nm is None:
        p = pm[(pm.nbeams == 16) & (pm.name == "mass")].groupby("stream")[["served", "overhead", "longest_miss_run"]].mean()
        b = rd("mass_rule/bootstrap_ci.csv").set_index(["nbeams", "name"]).loc[(16, "mass")]
        rows.append(f"{lab} & {f3(p.served.mean())} & {f3(p.served.min())} [{f2(b.lo)}, {f2(b.hi)}] & {f1(100*p.overhead.mean())} & {p.longest_miss_run.max():.0f}"); continue
    a = src.loc[nm]; lo, hi = ci("cb16", 0.0, nm)
    rows.append(f"{lab} & {f3(a['mean'])} & {f3(a.worst)} [{f2(lo)}, {f2(hi)}] & {f1(100*a.ovh)} & {a.run:.0f}")
write("table4_ops16", rows, "& \\multicolumn{2}{c}{Served coverage} & Ovh. & Longest\\\\\nScheme (16 beams) & mean & worst [95\\% CI] & (\\%) & miss run\\\\\n")
nd = rd("tuned_grids/bootstrap_needed.csv"); nd = nd[(nd.part == "cb16") & (nd.noise == 0)].set_index(["target", "family"])
for t in [0.85, 0.9]:
    num(f"16 beams, overhead needed for {t} (median [CI], point)", "; ".join(
        f"{fm} {nd.loc[(t, fm)]['median']:.1f} [{nd.loc[(t, fm)].lo:.1f}, {nd.loc[(t, fm)].hi:.1f}], {nd.loc[(t, fm)].point:.1f}"
        for fm in ["ours", "topK", "iqt", "hybrid", "grid"] if (t, fm) in nd.index))
pr16 = rd("tuned_grids/bootstrap_pairs.csv"); pr16 = pr16[(pr16.part == "cb16") & (pr16.a == "ours")].set_index("target")
num("16 beams, share of replicates where the online set is cheapest (0.85 / 0.88 / 0.9)",
    " / ".join(f"{100*pr16.loc[t].p_a_cheaper:.0f}%" for t in [0.85, 0.88, 0.9]))
num("16 beams, share of misses certified (%)", f"{100*z16[z16.name == 'QT-cert:0.05'].cert_recall.mean():.0f}")

# ---------- Table V: settings fixed before testing ----------
vp = rd("fixed_settings/chosen.csv"); vp = vp[vp.target == 0.9].set_index(["nbeams", "family", "xsel"])
def cell(nb, fam, xsel, col):
    if (nb, fam, xsel) not in vp.index: return "--"
    r = vp.loc[(nb, fam, xsel)]; w, o = r[f"worst_{col}"], r[f"ovh_{col}"]
    s_ = f"{w:.3f}"; s_ = f"\\textbf{{{s_}}}" if w >= 0.9 else s_
    return f"{s_}/{o:.1f}" + ("$^\\ast$" if (xsel == "lopo" and col == "X" and not r.val_ok) else "")
rows = []
for lab, fam in [("Grid, start 0", "grid0"), ("Grid, tuned$^\\dagger$", "tuned_grid"), ("Hybrid, start 0", "hybrid0"),
                 ("Hybrid, tuned$^\\dagger$", "tuned_hybrid"), ("GPS top-$K$", "topK"), ("Mass rule (no setting)", None),
                 ("IQT", "iqt"), ("Online set", "online")]:
    if fam is None:
        p = pm[(pm.name == "mass") & (pm.protocol != "A")].groupby(["nbeams", "protocol", "stream"])[["served", "overhead"]].mean()
        c = lambda nb, pr: f"{p.loc[(nb, pr)].served.min():.3f}/{100*p.loc[(nb, pr)].overhead.mean():.1f}"
        rows.append(f"{lab} & {c(64, 'E')} & {c(64, 'X')} & {c(64, 'X')} & {c(16, 'E')} & {c(16, 'X')} & {c(16, 'X')}"); continue
    rows.append(f"{lab} & " + " & ".join([cell(nb, fam, "lto", "E"), cell(nb, fam, "lto", "X"), cell(nb, fam, "lopo", "X")][i]
                                         for nb in [64, 16] for i in range(3)))
write("table5_fixed_settings", rows, "& Future drives & \\multicolumn{2}{c}{New session, chosen on} & Future drives & \\multicolumn{2}{c}{New session, chosen on}\\\\\n"
      "Family & (validation) & validation & other pairs & (validation) & validation & other pairs\\\\\n")
for (nb, fam, xsel), r in vp.iterrows():
    num(f"Table V choice {nb} beams {fam} {xsel}", f"future: {r.choice_E}; new session: {r.choice_X}")

# ---------- Table VI: probe noise ----------
nzs = [0.25, 0.5, 1.0]
def noisy(nm):
    if nm.startswith("tuned-"): d = tg["cb64n"]
    else: d = cn[cn.remeasure == 3]
    d = d[d.name == nm]
    return {z: d[d.noise == z] for z in nzs}
rows = []
for lab, nm in [("Exhaustive", "fixed:64"), ("Grid, every 4th$^\\dagger$", "tuned-uniform:4"), ("Grid, every 6th$^\\dagger$", "tuned-uniform:6"),
                ("Grid, every 8th$^\\dagger$", "tuned-uniform:8"), ("Top-1 + 8th$^\\dagger$", "tuned-hybrid:8"),
                ("IQT, $\\alpha{=}0.05$", "QT-sweep:0.05"), ("Online, $\\alpha{=}0.05$", "QT-cert:0.05")]:
    v = noisy(nm); lo, hi = ci("cb64n", 1.0, nm); o = v[1.0]
    m_ = {z: v[z].groupby("stream")[["served", "overhead"]].mean() for z in nzs}
    rows.append(f"{lab} & " + " & ".join(f"{m_[z].served.min():.2f}/{100*m_[z].overhead.mean():.0f}" for z in nzs) +
                f" & {lo:.2f}--{hi:.2f} & {o.worst_roll100.min():.2f}/{o.longest_miss_run.max():.0f}")
write("table6_noise", rows, "& \\multicolumn{3}{c}{Worst served / mean ovh.\\ (\\%)} & \\multicolumn{2}{c}{1 dB}\\\\\nScheme & 0.25 dB & 0.5 dB & 1 dB & 95\\% CI & win./run\\\\\n")
g0 = cn[(cn.name == "uniform:4") & (cn.remeasure == 3) & (cn.noise == 0.5)].groupby("stream").served.mean().min()
num("0.5 dB, grid every 4th from beam 0 (re-measured): worst", f"{g0:.2f}")
qn = cn[(cn.name == "QT-cert:0.05") & (cn.noise == 1.0)]
num("1 dB: share of certificates that are false (%)", f"{100*qn.cert_false_rate.mean():.0f}")
sv = [round(100 * cn[(cn.name == 'QT-sweep:0.05') & (cn.noise == z) & (cn.remeasure == 3)].groupby('stream').overhead.mean().mean())
      - round(100 * cn[(cn.name == 'QT-cert:0.05') & (cn.noise == z) & (cn.remeasure == 3)].groupby('stream').overhead.mean().mean()) for z in nzs]
num("overhead saved by certificates under noise (points)", f"{min(sv)}-{max(sv)}")
loss = []
for nm in cn.name.unique():
    a = cn[(cn.name == nm) & (cn.noise == 1.0)]
    if a.remeasure.nunique() < 2: continue
    w3 = a[a.remeasure == 3].groupby("stream").served.mean().min(); w0 = a[a.remeasure == 0].groupby("stream").served.mean().min()
    loss.append(w3 - w0)
num("1 dB: worst-stream loss without re-measurement", f"{min(loss):.2f}-{max(loss):.2f}")
nn = rd("tuned_grids/bootstrap_needed.csv"); nn = nn[(nn.part == "cb64n") & (nn.noise == 1.0) & (nn.target == 0.85) & (nn.family == "ours")]
if len(nn): num("1 dB: share of replicates where the online set never reaches 0.85", f"{100*nn.p_never.iloc[0]:.0f}%")

# ---------- streams and the validated settings ----------
s = rd("streams.csv"); num("test drives: future-drive / cross-session", f"{s[s.protocol == 'future-drive'].test_drives.min()}-{s[s.protocol == 'future-drive'].test_drives.max()} / "
                                                                   f"{s[s.protocol == 'cross-session'].test_drives.min()}-{s[s.protocol == 'cross-session'].test_drives.max()}")
num("test samples: future-drive / cross-session", f"{s[s.protocol == 'future-drive'].test_samples.min()}-{s[s.protocol == 'future-drive'].test_samples.max()} / "
                                                   f"{s[s.protocol == 'cross-session'].test_samples.min()}-{s[s.protocol == 'cross-session'].test_samples.max()}")
num("streams with fewer than 5 test drives", f"{(s.test_drives < 5).sum()}")
ch = rd("drift/choices.csv"); num("validated settings (left_out 0 = none)", ch.to_string(index=False).replace("\n", " | "))
lo = rd("leave_target_out/summary.csv"); num("largest change of an online scheme without the target scenario", f"{(lo.worst_left_out - lo.worst).abs().max():.3f}")

# ---------- Fig. 2 ----------
plt.rcParams.update({"font.size": 7, "font.family": "serif", "pdf.fonttype": 42})
fams = [("GPS top-$K$", lambda n, nb: n.startswith("fixed:") and n != f"fixed:{nb}", "#7a7a7a", "o", ":", "white"),
        ("Online set (ours)", lambda n, nb: n.startswith("QT-cert:"), "#c0392b", "D", "-", None),
        ("Oracle-calibrated set", lambda n, nb: n.startswith("QT-oracle:"), "#e87ba4", "s", "-.", "white"),
        ("Grid, start 0", lambda n, nb: n.startswith("uniform:"), "#1f4e9c", "v", "-", None),
        ("Top-1 + grid, start 0", lambda n, nb: n.startswith("hybrid:(1,"), "#1baf7a", "^", (0, (4, 1, 1, 1, 1, 1)), None)]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.05))
PMf = pm[pm.protocol != "A"].groupby(["nbeams", "name", "stream"])[["served", "overhead"]].mean().reset_index()
PMf = PMf.groupby(["nbeams", "name"]).agg(worst=("served", "min"), overhead=("overhead", "mean")).reset_index()
for ax, Ag, Tg, nb, title in [(axs[0], A64, T64, 64, "(a) 64 beams"), (axs[1], A16, T16, 16, "(b) 16 beams (every 4th)")]:
    for lab, sel, col, mk, ls, mfc in fams:
        a = Ag[[sel(n, nb) for n in Ag.index]].sort_values("ovh")
        ax.plot(100 * a.ovh, a.worst, color=col, marker=mk, ms=3.8, lw=1.3 if "ours" in lab else 0.9, ls=ls,
                mfc=mfc or col, mec=col, mew=0.8, label=lab)
    t_ = Tg[Tg.index.str.startswith("tuned-uniform:")].sort_values("ovh")
    ax.plot(100 * t_.ovh, t_.worst, color="#1f4e9c", marker="v", mfc="white", mew=0.8, ms=3.8, lw=0.9, ls="--", label="Grid, tuned start")
    p_ = PMf[(PMf.nbeams == nb) & (PMf.name == "mass")]
    ax.plot(100 * p_.overhead, p_.worst, ls="", marker="P", color="k", mfc="white", mew=0.8, ms=5.0, label="Mass rule")
    for nm, lab, mk in [("track:None", "Tracking", "x"), (f"fixed:{nb}", "_exhaustive", "*")]:
        if nm in Ag.index: ax.plot(100 * Ag.loc[nm, "ovh"], Ag.loc[nm, "worst"], ls="", marker=mk, color="k", ms=5.5, label=lab)
    ax.axhline(0.9, color="k", lw=0.5, ls="--"); ax.set_xscale("log"); ax.set_xticks([10, 20, 40, 100])
    ax.set_xticklabels(["10", "20", "40", "100"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Mean total overhead (%)", fontsize=7); ax.tick_params(labelsize=6.5); ax.grid(lw=0.3, color="#dddddd")
    ax.set_title(title, fontsize=7, pad=2); ax.set_ylim(0.25, 1.01)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axs[0].set_ylabel("Worst-stream served coverage", fontsize=7)
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.985), frameon=False, fontsize=5.7, handlelength=2.2,
           columnspacing=0.6, handletextpad=0.4, markerscale=1.0)
fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(f"{OUT}/fig_frontier.pdf", bbox_inches="tight"); plt.close(fig)

with open(f"{OUT}/numbers.txt", "w") as fh: fh.write("\n".join(NUM) + "\n")
print(f"tables, fig_frontier.pdf and numbers.txt written to {OUT}")
