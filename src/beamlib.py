# Data loading, splits and predictors for the DeepSense 6G scenarios 1-9 (pre-processed data of the
# Position-Beam-Prediction repository), and the small conformal helpers shared by all experiment scripts.
import glob, os, pickle, warnings, zlib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
NB = 64                                                 # codebook size; set to 16 for the decimated codebook
TARGETS = {"1dB": 10 ** (-1 / 10)}                      # good beam: power >= rho * best power
CERT = 10 ** (1 / 10)                                   # certificate: a probed beam >= 1 dB above the set's best


def load(D, s):
    """positions, 64-beam powers and drive index (seq_index) of scenario s"""
    g = lambda k: np.load(glob.glob(f"{D}/scenario{s}_{k}_1-*.npy")[0], allow_pickle=True)
    cal = glob.glob(f"{D}/scenario{s}_unit2_loc_cal_*.npy")
    loc = (np.load(cal[0], allow_pickle=True) if cal else g("unit2_loc")).astype(float)
    return loc, g("unit1_pwr_60ghz").astype(float), g("seq_index").squeeze().astype(int)


def splits(seq, rng):
    """A_random: random 60/20/20 split of the samples, as in Morais et al.
       E_blocked_future: train on the first 55% of the drives, calibrate on the next 25%, test on the last 20% in time order."""
    N = len(seq); u = np.unique(seq); k = len(u); p = rng.permutation(N)
    return {"A_random": (p[:int(.6 * N)], p[int(.6 * N):int(.8 * N)], np.sort(p[int(.8 * N):])),
            "E_blocked_future": (np.where(np.isin(seq, u[:int(.55 * k)]))[0],
                                 np.where(np.isin(seq, u[int(.55 * k):int(.8 * k)]))[0],
                                 np.where(np.isin(seq, u[int(.8 * k):]))[0])}


CACHE = os.environ.get("BEAM_MODEL_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "_models"))


def train_mlp(Z, y, tr, seed):
    """MLP of Morais et al. (ICC 2023): 3 x 256 ReLU, cross-entropy, Adam, initial learning rate 1e-2 reduced by 0.2 at
    epochs 20 and 40, batch 32, 60 epochs, inputs quantized to 200 bins; the epoch with the best top-1 accuracy on a
    random 15% hold-out of the training samples is kept (Morais et al. use a random validation split). The
    calibration data are never used to fit or select the model."""
    import copy
    tr = np.random.default_rng(seed).permutation(tr); n_ho = max(1, int(round(0.15 * len(tr))))
    fit_i, ho_i = tr[:-n_ho], tr[-n_ho:]
    m = MLPClassifier((256, 256, 256), solver="adam", learning_rate_init=1e-2, batch_size=32, alpha=0.0,
                      shuffle=True, random_state=seed)
    best, state = -1.0, None
    for ep in range(60):
        if ep in (20, 40): m._optimizer.learning_rate_init *= 0.2
        m.partial_fit(Z[fit_i], y[fit_i], classes=np.arange(NB))
        acc = (m.predict(Z[ho_i]) == y[ho_i]).mean()
        if acc > best: best, state = acc, (copy.deepcopy(m.coefs_), copy.deepcopy(m.intercepts_))
    m.coefs_, m.intercepts_ = state
    return m


def fit_predict(name, X, y, tr, others, seed, raw=False):
    lo, hi = X[tr].min(0), X[tr].max(0)                  # min-max fitted on train only
    Z = np.clip((X - lo) / (hi - lo + 1e-12), 0, 1)
    if name == "mlp":
        Z = np.round(Z * 199) / 199                        # 200 quantization bins, as in Morais et al.
        key = zlib.crc32(np.ascontiguousarray(tr).tobytes() + np.ascontiguousarray(y[tr]).tobytes() +
                         np.ascontiguousarray(X[tr]).tobytes() + repr((seed, NB)).encode())
        f = os.path.join(CACHE, f"mlp_{NB}_{seed}_{key:08x}.pkl")
        if os.path.exists(f):
            with open(f, "rb") as fh: m = pickle.load(fh)
        else:
            m = train_mlp(Z, y, np.asarray(tr), seed); os.makedirs(CACHE, exist_ok=True)
            tmp = f + f".{os.getpid()}"
            with open(tmp, "wb") as fh: pickle.dump(m, fh)
            os.replace(tmp, f)
    else:                                                  # kNN: 25 neighbours, distance-weighted, Euclidean
        m = KNeighborsClassifier(n_neighbors=25, weights="distance").fit(Z[tr], y[tr])
    prior = np.bincount(y[tr], minlength=NB) / len(tr); res = []
    for idx in others:
        P = np.zeros((len(idx), NB)); P[:, m.classes_] = m.predict_proba(Z[idx])
        res.append(P if raw else 0.98 * P + 0.01 * prior + 0.01 / NB)  # tiny smoothing: no zeros, no ties at 0 (raw: as predicted)
    return res


def good_mask(pwr, rho): return pwr >= rho * pwr.max(1, keepdims=True)
def ranks(P): return np.argsort(np.argsort(-P, 1), 1) + 1          # 1 = most likely beam


def scores(P, G, kind="rank"):
    """rank: best rank among the good beams (set = top-q beams); thr: 1 - highest probability among the good beams"""
    if kind == "thr": return 1 - np.where(G, P, 0).max(1)
    return np.where(G, ranks(P), NB + 1).min(1).astype(float)


def qhat(sc, a):
    """split-conformal quantile of the calibration scores at level 1 - a"""
    n = len(sc)
    if a <= 0: return np.inf
    if a >= 1: return -np.inf
    lev = np.ceil((n + 1) * (1 - a)) / n
    return np.inf if lev > 1 else np.quantile(sc, lev, method="higher")


def static_set(P, G, sc_cal, alpha, kind="rank"):
    """coverage and mean size of the static split-conformal set on a stream"""
    q = qhat(sc_cal, alpha); S = (ranks(P) <= q) if kind == "rank" else ((1 - P) <= q)
    return (S & G).any(1).mean(), S.sum(1).mean()


def fit_temperature(P, y):
    """temperature that minimizes the negative log-likelihood of the labels y"""
    L = np.log(np.clip(P, 1e-12, None)); best = (np.inf, 1.0)
    for T in np.exp(np.linspace(np.log(0.05), np.log(10), 200)):
        Z = L / T; Z -= Z.max(1, keepdims=True)
        best = min(best, (-(Z[np.arange(len(y)), y] - np.log(np.exp(Z).sum(1))).mean(), T))
    return best[1]
