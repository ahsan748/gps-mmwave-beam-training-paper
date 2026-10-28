# Reproduces every result, table and figure of the paper (about 2 h on 2 CPU cores; trained models are cached).
# Run from the repository root, with the Position-Beam-Prediction data folder next to this file:
#   python run_all.py                       full run into results/ and outputs/
#   python run_all.py --quick               smoke test on three streams and one seed (a few minutes)
#   python run_all.py --quick --dir test    write results and outputs under test/ instead
import os, subprocess, sys

root = os.path.dirname(os.path.abspath(__file__)); src = os.path.join(root, "src")
data = os.path.join(root, "Position-Beam-Prediction", "Gathered_data_DEV")
base = os.path.join(root, sys.argv[sys.argv.index("--dir") + 1]) if "--dir" in sys.argv else root
res, out = os.path.join(base, "results"), os.path.join(base, "outputs")
R = {k: os.path.join(res, k) for k in ["drift", "main", "coarse_noise", "grid_start", "grid_start_tuned", "tuned_grids",
                                       "mass_rule", "leave_target_out", "fixed_settings", "eta"]}
if not os.path.isdir(data):
    sys.exit(f"Data folder not found: {data}\nFirst run: git clone --depth 1 https://github.com/jmoraispk/Position-Beam-Prediction.git")
for d in [out, *R.values()]: os.makedirs(d, exist_ok=True)
os.environ.setdefault("BEAM_MODEL_CACHE", os.path.join(res, "_models"))
quick = "--quick" in sys.argv
q = ["--streams", "E4", "E5", "X8-9", "--seeds", "1", "--sweep-seeds", "1"] if quick else []
q1 = ["--seeds", "1"] if quick else []
nb = "200" if quick else "1000"
py = lambda f, *a: [sys.executable, os.path.join(src, f), *a]
steps = [py("validate_settings.py", "--data", data, "--out", R["drift"], *q1),               # settings (see settings.py)
         py("drift_runs.py", "--data", data, "--out", R["drift"], *q),                        # Table I, kNN
         py("main_runs.py", "--data", data, "--part", "cb64", "--out", R["main"], *q),       # Tables II and III
         py("main_runs.py", "--data", data, "--part", "cb16", "--out", R["main"], *q),
         py("bootstrap_main.py", R["main"], nb),
         py("coarse_noise_runs.py", "--data", data, "--part", "cb16", "--out", R["coarse_noise"], *q),   # Table IV
         py("coarse_noise_runs.py", "--data", data, "--part", "cb64n", "--out", R["coarse_noise"], *q),  # Table VI
         py("bootstrap_coarse_noise.py", R["coarse_noise"], nb),
         py("grid_start_runs.py", "--data", data, "--out", R["grid_start"], *q),
         py("grid_start_tuning.py", "--data", data, "--grid-start", R["grid_start"], "--out", R["grid_start_tuned"], *q1),
         *[py("tuned_grid_runs.py", "--data", data, "--part", p, "--out", R["tuned_grids"], *q) for p in ["cb64", "cb16", "cb64n"]],
         py("bootstrap_tuned.py", R["main"], R["coarse_noise"], R["tuned_grids"], nb),
         py("codebook_overlap.py", data, os.path.join(R["coarse_noise"], "overlap.csv")),
         py("mass_rule_runs.py", "--data", data, "--out", R["mass_rule"], *q, "--boot", nb),
         py("leave_target_out_runs.py", "--data", data, "--choices", os.path.join(R["drift"], "choices.csv"),
            "--out", R["leave_target_out"], *q1, *(["--sweep-seeds", "1"] if quick else [])),
         py("leave_target_out_summary.py", res),
         py("fixed_settings.py", "--data", data, "--results", res, "--out", R["fixed_settings"], *q1),  # Table V
         py("eta_sensitivity.py", "--data", data, "--out", R["eta"], *q),
         py("stream_counts.py", "--data", data, "--out", os.path.join(res, "streams.csv")),
         py("make_paper.py", res, out),                                                       # tables, Fig. 2, numbers.txt
         py("make_fig_example.py", "--data", data, "--out", out)]                             # Fig. 1
for s in steps:
    print(">", " ".join(os.path.relpath(x, root) if os.path.isabs(x) else x for x in s[1:]), flush=True)
    subprocess.run(s, check=True, cwd=root)
print("done: results in", os.path.relpath(res, root), "and tables/figures in", os.path.relpath(out, root))
