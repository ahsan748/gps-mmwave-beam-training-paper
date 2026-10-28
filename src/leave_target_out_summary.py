# Worst-stream served coverage and mean overhead of the online schemes as reported (eta and M chosen on the
# validation drives of all nine scenarios) and with eta/M chosen without the target scenario of each cross-session
# stream (leave_target_out_runs.py). The grid and hybrid choices do not change without the target (results/drift/choices.csv);
# validation-chosen grids, hybrids and alpha are handled by fixed_settings.py. Usage: python leave_target_out_summary.py results
import sys
import pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "results"
src = {"cb64": (pd.read_csv(f"{R}/main/results_cb64.csv"), "main_runs"),
       "cb16": (pd.read_csv(f"{R}/coarse_noise/results_cb16.csv"), "coarse_noise_runs"),
       "cb64n": (pd.read_csv(f"{R}/coarse_noise/results_cb64n.csv"), "coarse_noise_runs")}
L = pd.read_csv(f"{R}/leave_target_out/results_leave_target_out.csv")


def per_stream(df, name, noise, rm):
    d = df[(df.name == name) & (df.noise == noise) & (df.remeasure == rm)]
    return d.groupby("stream")[["served", "overhead"]].mean()


rows = []
for part, nz, rm, names in [("cb64", 0.0, 0, ["QT-cert:0.05", "QT-cert:0.1", "QT-oracle:0.05", "QT-oracle:0.1", "QT-sweep:0.05", "QT-sweep:0.1"]),
                            ("cb16", 0.0, 0, ["QT-cert:0.05", "QT-cert:0.1", "QT-oracle:0.05", "QT-sweep:0.05", "QT-sweep:0.1"]),
                            *[("cb64n", z, 3, ["QT-cert:0.05", "QT-sweep:0.05"]) for z in [0.25, 0.5, 1.0]]]:
    df, s = src[part]
    for nm in names:
        a = per_stream(df, nm, nz, rm); b = a.copy()
        d = L[(L.source == s) & (L.part == part) & (L.name == nm) & (L.noise == nz) & (L.remeasure == rm)].groupby("stream")[["served", "overhead"]].mean()
        d = d[d.index.isin(b.index)]; b.loc[d.index] = d.values
        rows.append(dict(part=part, noise=nz, scheme=nm, worst=a.served.min(), at=a.served.idxmin(), ovh=100 * a.overhead.mean(),
                         worst_left_out=b.served.min(), at_left_out=b.served.idxmin(), ovh_left_out=100 * b.overhead.mean()))
O = pd.DataFrame(rows); print(O.round(3).to_string(index=False)); O.to_csv(f"{R}/leave_target_out/summary.csv", index=False)
