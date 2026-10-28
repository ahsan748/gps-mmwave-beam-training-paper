# When Does GPS Save mmWave Beam-Training Overhead?

Code, results and scripts for the paper

> A. Ali, A. Afzal, et al., "When Does GPS Save mmWave Beam-Training Overhead? A Reliability-Calibrated
> Evaluation on Real Drives," submitted to *IEEE Communications Letters*, 2026.

The paper replays real 60 GHz beam measurements from DeepSense 6G (scenarios 1–9) and asks how much beam-training
overhead position-aided beam prediction saves once reliability is measured on later drives and in new sessions,
every probe is counted, and the GPS-free baselines are tuned as well. It compares split-conformal and
probability-mass beam sets, an online quantile tracker that uses hill-climb miss certificates and rare full sweeps,
coarse grids with a hill-climb, GPS + grid hybrids and beam tracking, on the 64-beam codebook and on a 16-beam
decimation, with and without probe noise.

## Data

The data are not redistributed here. Clone the pre-processed DeepSense 6G data of Morais et al. (ICC 2023) into the
repository root:

    git clone --depth 1 https://github.com/jmoraispk/Position-Beam-Prediction.git

`src/check_time_order.py` checks that the recorded sample order of a scenario matches its UTC timestamps.

## Requirements

Python 3.10 or later (tested with 3.13):

    pip install -r requirements.txt

## Reproduce

    python run_all.py            # everything, about 2 h on 2 CPU cores
    python run_all.py --quick    # smoke test on three streams and one seed, a few minutes
    python run_all.py --quick --dir test   # the same, writing to test/results and test/outputs

All seeds are fixed. Trained models are cached in `results/_models` (about 1.3 GB, not in the repository), so a
second run is much faster. The full run rewrites `results/` and `outputs/`.

## Where each part of the paper comes from

| Paper item | Script(s) | Output |
|---|---|---|
| Design choices (step, sweep rate, grid, hybrid) | `validate_settings.py` | `results/drift/validation.csv`, `choices.csv`, fixed in `src/settings.py` |
| Table I (drift) | `drift_runs.py`, `mass_rule_runs.py` | `outputs/table1_drift.tex` |
| Table II (feedback) | `main_runs.py` | `outputs/table2_feedback.tex` |
| Table III (64 beams) | `main_runs.py`, `grid_start_runs.py`, `grid_start_tuning.py`, `tuned_grid_runs.py`, `bootstrap_main.py` | `outputs/table3_ops64.tex` |
| Table IV (16 beams) | `coarse_noise_runs.py`, `tuned_grid_runs.py`, `bootstrap_coarse_noise.py` | `outputs/table4_ops16.tex` |
| Table V (settings fixed before testing) | `fixed_settings.py` | `outputs/table5_fixed_settings.tex` |
| Table VI (probe noise) | `coarse_noise_runs.py`, `tuned_grid_runs.py`, `bootstrap_tuned.py` | `outputs/table6_noise.tex` |
| Fig. 1 | `make_fig_example.py` | `outputs/fig0_examples.pdf` |
| Fig. 2 | `make_paper.py` | `outputs/fig_frontier.pdf` |
| Numbers quoted in the text | `make_paper.py` | `outputs/numbers.txt` |
| Overhead needed, share of replicates | `bootstrap_*.py` | `results/*/bootstrap*.csv` |
| Choices without the target scenario | `leave_target_out_runs.py`, `leave_target_out_summary.py` | `results/leave_target_out/summary.csv` |
| Step-size sensitivity | `eta_sensitivity.py` | `results/eta/summary.csv` |
| Test-stream sizes and drive counts | `stream_counts.py` | `results/streams.csv` |
| Codebook overlap (Sec. II) | `codebook_overlap.py` | `results/coarse_noise/overlap.csv` |

Results without re-measurement under noise are in `results/coarse_noise/results_cb64n.csv` (`remeasure = 0`).
Per-run results of every scheme and stream are in the `results_*.csv` files; the per-sample traces used by the
bootstrap are written as `traces_*.pkl` and are not included in the repository.

## Code

- `src/beamlib.py`: data loading, splits, the MLP and kNN predictors, conformal helpers.
- `src/beamsim.py`: the serving simulator (sets, grids, hybrids, tracking, the hill-climb, probe noise and
  re-measurement, and the online trackers).
- `src/streams.py`: the nine future-drive and six cross-session test streams.
- `src/settings.py`: the settings chosen on the validation drives.
- `src/*_runs.py`: the experiments; `src/bootstrap_*.py`: paired drive-level bootstrap.
- `src/make_paper.py`, `src/make_fig_example.py`: tables, figures and the numbers used in the text.

## License

MIT, see `LICENSE`. The DeepSense 6G data are subject to their own terms.
