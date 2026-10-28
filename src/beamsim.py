# Beam-set serving simulator.
# Every policy can use the same hill-climb as a serving aid, and every probe is counted in the overhead.
# Decisions use MEASURED powers (optionally with per-probe dB noise); evaluation uses the TRUE powers.
#
# Policies (pol):
#   ("fixed", K)         top-K beams of the predictor
#   ("pmass", (tau, T))  per-sample smallest top set whose predicted probability (temperature T) reaches tau
#   ("static", None)     split-conformal top-q (q from calibration, rank score)
#   ("uniform", d)       GPS-free: probe every d-th beam (0, d, 2d, ...); ("uniform", (d, o)) starts at beam o
#   ("hybrid", (K, d))   GPS top-K united with every d-th beam; (K, d, o) starts the grid at beam o
#   ("track", None)      GPS-free: probe the beam served at the previous sample of the same drive; full sweep at drive start
#   ("trackgrid", (N,d)) GPS-free: tracking, plus a coarse grid (every d-th beam) on every N-th sample (NR P1 + P3 style)
#   ("QT", fb)           quantile tracking of q, feedback fb in {"sweep", "cert", "served", "oracle"}
#   ("ACI", fb)          adaptive conformal inference (level alpha_t), fb in {"sweep", "cert"}
#   ("DtACI", "sweep")   DtACI on swept scores
#   ("RW", W)            split conformal on the last W revealed scores (calibration scores, then swept scores)
# Feedback:
#   sweep  : e_t = c0_t + M xi_t (m_t - c0_t), c0_t = 0 (set never empty here)
#   cert   : e_t = c_t + M xi_t (m_t - c_t),  c_t = hill-climb certificate
#   served : e_t = M xi_t sm_t, sm_t = served miss (set + climb), i.e. tracks SERVED coverage, sweeps only
#   oracle : e_t = m_t (true set miss, every sample, free; not realisable)
import numpy as np
from beamlib import ranks, qhat
DTACI_GAMMAS = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]   # DtACI step sizes

CERT_DB = 1.0


def climb(p, probes_set, start, return_probes=False):
    """Uphill walk over b+-1 from beam `start` on measured powers p. Returns (end beam, new probes, steps[, probe set])."""
    nbeams = len(p); probes = set(probes_set); b = start; new = 0; steps = 0
    while True:
        nb = [j for j in (b - 1, b + 1) if 0 <= j < nbeams]
        add = [j for j in nb if j not in probes]; new += len(add); probes.update(add)
        j = max(nb, key=lambda j: p[j])
        if p[j] > p[b]: b = j; steps += 1
        else: return (b, new, steps, probes) if return_probes else (b, new, steps)


def simulate(Pt, P, rho, sc_cal, alpha, pol, M=None, rng=None, noise_db=0.0, climb_aid=True,
             seq=None, eta=0.5, gamma=0.05, rec_traces=False, grid=0, remeasure=0):
    """grid > 0: every set is united with the coarse grid {0, grid, 2*grid, ...} (GPS set as a complement).
    remeasure = r > 0: noise-aware refinement for every scheme: after the climb, the r strongest probed beams are
    measured once more (fresh noise, r extra probes), the two looks are averaged in dB, and the best is served."""
    pname, par = pol
    NB = P.shape[1]                                              # codebook size (64, or 16 for the decimated codebook)
    n = len(P); Rk = ranks(Pt)
    rng = rng or np.random.default_rng(0)
    nrng = np.random.default_rng(rng.integers(1 << 31))
    Pm = P * 10 ** (noise_db * nrng.standard_normal(P.shape) / 10) if noise_db > 0 else P   # measured powers
    Pm2 = P * 10 ** (noise_db * nrng.standard_normal(P.shape) / 10) if (noise_db > 0 and remeasure) else Pm  # 2nd look
    G = P >= rho * P.max(1, keepdims=True)                     # truth
    Gm = Pm >= rho * Pm.max(1, keepdims=True)                  # what a sweep reveals
    sc_m = np.where(Gm, Rk, NB + 1).min(1).astype(float)       # revealed rank score
    online = pname in ("QT", "ACI", "DtACI", "RW")
    if pname in ("track", "trackgrid"):
        sw = np.r_[True, seq[1:] != seq[:-1]]                  # full sweep at every drive start
    else:
        sw = (rng.random(n) < 1 / M) if (online and M and par != "oracle") else np.zeros(n, bool)
    q = qhat(np.asarray(sc_cal), alpha) if pname in ("static", "QT") else None
    a = alpha; pool = list(sc_cal)
    if pname == "pmass":                                         # probability-mass rule of Morais et al. (T = 1) or Hiam et al. (fitted T)
        tau, T = par; Zp = np.clip(Pt, 1e-12, None) ** (1.0 / T); Zp = Zp / Zp.sum(1, keepdims=True)
        Kt = (np.cumsum(-np.sort(-Zp, 1), 1) < tau).sum(1) + 1
    if pname == "RW": pool = pool[-par:]
    if pname == "DtACI":
        g = np.array(DTACI_GAMMAS); KK = len(g); I = 20; sig = 1 / (2 * I)
        et = np.sqrt(3 / I) * np.sqrt((np.log(KK * I) + 2) / ((1 - alpha) ** 2 * alpha ** 3 + alpha ** 2 * (1 - alpha) ** 3))
        ak = np.full(KK, alpha); w = np.ones(KK) / KK
    prev = None; tpos = 0
    set_cov, srv_cov, R, probes_t, ks, steps_l = [], [], [], [], [], []
    extras, full_l, keep_idx, mm_sum = [], [], [], 0.0
    e_sum = m_sum = 0.0; n_cert = n_cert_false = n_miss = n_cert_true = 0; sm_all = []
    for t in range(n):
        p, pm = P[t], Pm[t]
        # ---- the set ----
        if pname == "fixed": S = Rk[t] <= par
        elif pname == "pmass": S = Rk[t] <= Kt[t]                    # smallest top set with predicted mass >= tau
        elif pname == "uniform":                                  # par = d, or (d, offset)
            d_, o_ = (par, 0) if np.isscalar(par) else par; S = (np.arange(NB) % d_) == o_
        elif pname == "hybrid":                                   # GPS top-K + coarse grid; par = (K, d) or (K, d, offset)
            o_ = par[2] if len(par) > 2 else 0; S = (Rk[t] <= par[0]) | ((np.arange(NB) % par[1]) == o_)
        elif pname in ("track", "trackgrid"):
            tpos = 0 if (t == 0 or seq[t] != seq[t - 1]) else tpos + 1
            S = np.zeros(NB, bool)
            if pname == "trackgrid" and tpos % par[0] == 0: S = (np.arange(NB) % par[1]) == 0
            elif prev is not None: S[prev] = True
        else:
            if pname in ("ACI", "DtACI", "RW"):
                if pname == "DtACI": a = float(w @ ak)
                q = qhat(np.asarray(pool if pname != "RW" else pool[-par:]), a)
            S = Rk[t] <= max(q, 1.0)                              # never empty
        if grid: S = S | ((np.arange(NB) % grid) == 0)
        k = int(S.sum())
        hit = bool((S & G[t]).any())
        bS = int(np.flatnonzero(S)[np.argmax(pm[S])]) if k else int(np.argmax(pm))
        # ---- serving (hill-climb aid on measured powers) ----
        if climb_aid and k:
            b_end, extra, steps, prb = climb(pm, np.flatnonzero(S), bS, return_probes=True)
        else:
            b_end, extra, steps, prb = bS, 0, 0, set(np.flatnonzero(S))
        cert = float(pm[b_end] >= 10 ** (CERT_DB / 10) * pm[bS]) if (climb_aid and k) else 0.0
        b_srv_set = b_end
        if remeasure and k:                                      # noise-aware refinement (same for every scheme)
            cand = sorted(prb, key=lambda j: -pm[j])[:remeasure]
            avg = {j: 0.5 * (10 * np.log10(pm[j]) + 10 * np.log10(Pm2[t][j])) for j in cand}
            b_srv_set = max(avg, key=avg.get); extra += len(cand)
        served_ok = bool(p[b_srv_set] >= rho * p.max())
        mm_sum += float(not (S & Gm[t]).any())
        m = float(not hit)
        if sw[t]:                                                # full sweep: serve measured best
            probes_t.append(NB); b_srv = int(np.argmax(pm))
            m_rev = float(not (S & Gm[t]).any()); sm_rev = float(not (pm[b_end] >= rho * pm.max()))
        else:
            probes_t.append(k + extra); b_srv = b_srv_set
            set_cov.append(1 - m); srv_cov.append(float(served_ok)); R.append(p[b_srv_set] / p.max())
            ks.append(k); steps_l.append(steps); sm_all.append(1 - served_ok); extras.append(extra)
            full_l.append(k == NB); keep_idx.append(t)
            n_miss += int(m); n_cert += int(cert); n_cert_false += int(cert and not m); n_cert_true += int(cert and m)
        prev = b_srv
        # ---- feedback and update ----
        if online:
            xi = float(sw[t]); Mw = M or 0                      # no sweeps -> no correction term
            if par == "oracle": e = m
            elif par == "served": e = Mw * xi * (sm_rev if xi else 0.0)
            elif par == "cert": e = cert + Mw * xi * ((m_rev if xi else 0.0) - cert)
            else: e = Mw * xi * (m_rev if xi else 0.0)
            if pname == "QT":
                q = max(q + eta * (e - alpha), 1.0)              # projection from below only: set >= 1 beam
                e_sum += e; m_sum += float(not (S & Gm[t]).any())   # Prop. 1 is about the miss a sweep reveals
            elif pname == "ACI":
                a = a + gamma * (alpha - e)
            elif pname == "DtACI" and xi:
                ref = np.asarray(pool); beta = (np.sum(ref >= sc_m[t]) + 1) / (len(ref) + 1)
                loss = np.where(beta < ak, (1 - alpha) * (ak - beta), alpha * (beta - ak))
                wb = w * np.exp(-et * loss); w = (1 - sig) * wb / wb.sum() + sig / KK
                ak = ak + g * (alpha - (beta < ak).astype(float))
            if xi and pname in ("ACI", "DtACI", "RW"): pool.append(sc_m[t])
    srv = np.array(srv_cov); pr = np.array(probes_t) / NB
    roll = np.convolve(srv, np.ones(100) / 100, mode="valid") if len(srv) >= 100 else np.array([srv.mean()])
    runs, cur = 0, 0
    for v in sm_all:
        cur = cur + 1 if v else 0; runs = max(runs, cur)
    out = dict(set_cov=np.mean(set_cov), served=srv.mean(), overhead=pr.mean(), ovh_p50=np.median(pr),
               ovh_p95=np.percentile(pr, 95), ovh_max=pr.max(), R=np.mean(R), outage=np.mean(np.array(R) < 0.95),
               worst_roll100=roll.min(), longest_miss_run=runs, size=np.mean(ks), climb_steps=np.mean(steps_l),
               climb_steps_max=(max(steps_l) if steps_l else 0), n_test=n, n_sweeps=int(sw.sum()),
               cert_recall=(n_cert_true / n_miss if n_miss else np.nan),
               cert_false_rate=(n_cert_false / n_cert if n_cert else np.nan),
               loss_db=float(np.mean(-10 * np.log10(np.maximum(np.array(R), 1e-12)))),
               outage3db=float(np.mean(np.array(R) < 10 ** (-0.3))), climb_probes=float(np.mean(extras)),
               frac_full=float(np.mean(full_l)))
    if pname == "QT":
        out.update(eps_T=e_sum / n - alpha, delta_T=(m_sum - e_sum) / n)
    if rec_traces:
        return out, dict(served=srv, keep=np.array(keep_idx), probes=np.array(probes_t))
    return out
