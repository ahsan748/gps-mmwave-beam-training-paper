# Overlap of adjacent beams in the 64-beam codebook and in its every-4th-beam decimation (16 beams), all nine scenarios.
# Reports the median power of the stronger neighbour of the best beam (dB, relative to the best) and the mean number
# of beams within 1 dB of the best. Usage: python codebook_overlap.py Position-Beam-Prediction/Gathered_data_DEV results/coarse_noise/overlap.csv
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import beamlib as B
D = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None; rows = []; P = np.concatenate([B.load(D, s)[1] for s in range(1, 10)])
for nb in [64, 16]:
    p = P[:, ::64 // nb]; b = p.argmax(1); best = p[np.arange(len(p)), b]
    lo = np.where(b > 0, p[np.arange(len(p)), np.maximum(b - 1, 0)], 0); hi = np.where(b < nb - 1, p[np.arange(len(p)), np.minimum(b + 1, nb - 1)], 0)
    nbr = 10 * np.log10(np.maximum(np.maximum(lo, hi), 1e-30) / best)
    print(f"{nb} beams: stronger neighbour median {np.median(nbr):.2f} dB | beams within 1 dB: {(p >= best[:, None] * 10**-0.1).sum(1).mean():.1f} "
          f"| samples {len(p)}")
    rows.append(dict(nbeams=nb, neighbour_db=np.median(nbr), within1db=(p >= best[:, None] * 10**-0.1).sum(1).mean()))
if OUT: __import__("pandas").DataFrame(rows).to_csv(OUT, index=False)
