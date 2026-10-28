# Leave-target-out check for the cross-session streams. The step size eta and sweep rate M were chosen on the
# validation drives of all nine scenarios, which include the target scenario of each cross-session stream.
# validate_settings.py also repeats the choice with each target scenario left out (results/drift/choices.csv).
# This script reruns every online policy of main_runs.py and coarse_noise_runs.py on the cross-session streams
# whose eta/M change that way, with the same seeds as before.
# Usage: python leave_target_out_runs.py --data Position-Beam-Prediction/Gathered_data_DEV --choices results/drift/choices.csv --out results/leave_target_out
import argparse, os, sys
import pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beamsim as S
CHANGED = {}


def load_choices(path):
    C = pd.read_csv(path).set_index("left_out"); base = (C.loc[0, "eta"], int(C.loc[0, "M"])); out = {}
    for a, b in [(1, 2), (2, 1), (3, 4), (4, 3), (8, 9), (9, 8)]:
        c = (C.loc[b, "eta"], int(C.loc[b, "M"]))
        if c != base: out[f"X{a}-{b}"] = c
    return out


def job(D, module, part, stream, seed, sweep_seeds, eta, M):
    import importlib
    m = importlib.import_module(module); orig = S.simulate
    def sim(*a, **kw):
        pol = a[5] if len(a) > 5 else kw["pol"]
        if pol[0] in ("QT", "ACI"):
            kw["eta"] = eta
            if kw.get("M"): kw["M"] = M
        return orig(*a, **kw)
    m.S.simulate = sim
    try:
        rows, _ = m.job(D, part, stream, seed, sweep_seeds)
    finally:
        m.S.simulate = orig
    return [dict(r, source=module, eta_left_out=eta, M_left_out=M) for r in rows if r["name"].split(":")[0] in
            ("QT-cert", "QT-oracle", "QT-sweep", "QT-certonly", "ACI-cert")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results/leave_target_out")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--sweep-seeds", type=int, default=4)
    ap.add_argument("--choices", default="results/drift/choices.csv")
    ap.add_argument("--streams", nargs="*", default=None); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); CHANGED.update(load_choices(a.choices))
    print("eta/M without the target scenario:", CHANGED or "unchanged for every stream")
    if a.streams is None: a.streams = list(CHANGED)
    tasks = [(mod, part, st, sd) for mod, parts in [("main_runs", ["cb64", "cb16"]), ("coarse_noise_runs", ["cb16", "cb64n"])]
             for part in parts for st in a.streams if st in CHANGED for sd in range(a.seeds)]
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(job)(a.data, mod, part, st, sd, a.sweep_seeds, *CHANGED[st]) for mod, part, st, sd in tasks)
    df = pd.DataFrame([r for rows in res for r in rows], columns=["source", "part", "stream", "seed", "noise", "remeasure", "name",
                      "set_cov", "served", "overhead", "eta_left_out", "M_left_out"] if not tasks else None)
    df.to_csv(f"{a.out}/results_leave_target_out.csv", index=False)
    if not len(df): return
    print(df.groupby(["source", "part", "stream", "noise", "remeasure", "name"])[["set_cov", "served", "overhead"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
