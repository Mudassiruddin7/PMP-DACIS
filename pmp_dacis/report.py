"""Aggregation, paired statistics, LaTeX tables, figures and the numbers file used by the paper.

Every number that appears in the paper is read from the saved runs by this module. Seeds (class partitions) are the
replicates: tables give mean and sample standard deviation over seeds, paired differences give a t-interval over the
per-seed differences (episodes are shared between methods inside a seed).
"""
import glob
import json
import os

import numpy as np
from scipy import stats as sps

CRIT_LABEL = {
    "random": "Random", "l1": r"Filter $\ell_1$", "bn_gamma": r"BN scale $|\gamma|$", "taylor": "Taylor",
    "G": r"Gradient $\mathcal{G}$", "V": r"Variance $\mathcal{V}$", "D": r"Fisher $\mathcal{D}$",
    "GV": r"$\mathcal{G}{+}\mathcal{V}$", "GD": r"$\mathcal{G}{+}\mathcal{D}$", "VD": r"$\mathcal{V}{+}\mathcal{D}$",
    "dacis": "DACIS",
}
SCORES = ["random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"]
PV, ES13, ES3 = "plantvillage", "eurosat13", "eurosat3"


class Results:
    def __init__(self, root):
        self.root = root
        self.runs = {}
        for p in glob.glob(os.path.join(root, "runs", "*", "seed*", "*.json")):
            r = json.load(open(p))
            self.runs[(r["variant"], r["seed"], r["spec"]["id"])] = r
        self._npz = {}
        self.host = json.load(open(os.path.join(root, "host.json"))) if os.path.exists(os.path.join(root, "host.json")) else {}
        tp = os.path.join(root, "tuning_m_lr.json")
        self.tuning = json.load(open(tp)) if os.path.exists(tp) else None

    # ---- access ---------------------------------------------------------------------------
    def seeds(self, variant):
        return sorted({k[1] for k in self.runs if k[0] == variant})

    def has(self, variant, id_):
        ss = self.seeds(variant)
        return bool(ss) and all((variant, s, id_) in self.runs for s in ss)

    def seeds_with(self, variant, *ids):
        return [s for s in self.seeds(variant) if all((variant, s, i) in self.runs for i in ids)]

    def pid(self, variant, c):
        """Run id of score c under the plain prune-and-recover schedule (two naming generations)."""
        for cand in (f"A2_{c}_P_30", f"B_{c}_P_30"):
            if self.has(variant, cand):
                return cand
        return None

    def vals(self, variant, id_, key, getter=None):
        out = []
        for s in self.seeds_with(variant, id_):
            r = self.runs[(variant, s, id_)]
            try:
                out.append(getter(r) if getter else r["acc"][key]["mean"])
            except KeyError:
                continue
        return np.array(out, dtype=float)

    def ms(self, variant, id_, key, getter=None):
        if id_ is None:
            return None
        v = self.vals(variant, id_, key, getter)
        if len(v) == 0:
            return None
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    def eff(self, variant, id_, field, sub=None):
        if id_ is None:
            return None
        out = []
        for s in self.seeds_with(variant, id_):
            v = self.runs[(variant, s, id_)]["eff"].get(field)
            if v is None:
                continue
            out.append(v[sub] if sub else v)
        return float(np.mean(out)) if out else None

    def arr(self, variant, seed, id_, key):
        p = os.path.join(self.root, "runs", variant, f"seed{seed}", f"{id_}.npz")
        if p not in self._npz:
            self._npz[p] = np.load(p)
        return self._npz[p][key].astype(np.float64)

    def paired(self, variant, a, b, key):
        """a - b in accuracy points. Replicates are seeds: mean of the per-seed differences with a 95% t-interval.
        `wins` counts seeds in which a beats b; `ep_ci` is the half-width of the episode-level bootstrap interval
        (sampling error of the 2000 test episodes only, reported for completeness)."""
        if a is None or b is None:
            return None
        seeds = self.seeds_with(variant, a, b)
        if len(seeds) < 2:
            return None
        d = np.array([self.runs[(variant, s, a)]["acc"][key]["mean"] - self.runs[(variant, s, b)]["acc"][key]["mean"]
                      for s in seeds])
        n = len(d)
        sd = float(d.std(ddof=1))
        half = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        return {"mean": float(d.mean()), "lo": float(d.mean() - half), "hi": float(d.mean() + half), "sd": sd,
                "per_seed": d.tolist(), "wins": int((d > 0).sum()), "n": n}


# ---- formatting ---------------------------------------------------------------------------------
def pm(ms, d=1, bold=False):
    if ms is None:
        return "--"
    m, s = ms
    txt = f"{m:.{d}f}"
    if bold:
        txt = r"\mathbf{" + txt + "}"
    return f"${txt}" + r"_{\pm" + f"{s:.{d}f}" + "}$"


def sgn(x, d=1):
    return f"{x:+.{d}f}".replace("-", "{-}")


def pd_cell(p, d=1):
    if p is None:
        return "--"
    return f"${sgn(p['mean'], d)}$ " + r"{\scriptsize[" + f"{sgn(p['lo'], d)},{sgn(p['hi'], d)}" + "]}"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def mparams(R, variant, id_):
    p = R.eff(variant, id_, "params")
    return None if p is None else p / 1e6


def _bestmask(cells):
    vals = [c[0] if c is not None else None for c in cells]
    ok = [v for v in vals if v is not None]
    top = max(ok) if ok else None
    return [v is not None and v == top for v in vals]


# ---- tables -------------------------------------------------------------------------------------
def table_scores(R):
    """Eight channel scores at 30% of the parameters under two schedules (P: prune then recover; PMP)."""
    cols = [(PV, "novel_5", "P"), (PV, "novel_5", "PMP"), (PV, "plantdoc_5", "P"), (PV, "plantdoc_5", "PMP"),
            (ES13, "novel_5", "P"), (ES13, "novel_5", "PMP")]
    rows = []
    for c in SCORES:
        rows.append((CRIT_LABEL[c], {(v, k, s): R.ms(v, R.pid(v, c) if s == "P" else f"A_{c}_PMP_30", k) for v, k, s in cols}))
    refs = []
    for lab, ids in (("Dense, S0 only", "dense_ce"), ("Dense, S0+R+M+R", "dense_rmr")):
        refs.append((lab, {(v, k, s): R.ms(v, ids, k) for v, k, s in cols if s == "P"}))
    best = {col: _bestmask([r[1][col] for r in rows]) for col in cols}
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{PlantVillage} & \multicolumn{2}{c}{PlantDoc} & \multicolumn{2}{c}{EuroSAT-MS} \\",
         r" & \multicolumn{2}{c}{5-shot, novel} & \multicolumn{2}{c}{5-shot, field} & \multicolumn{2}{c}{5-shot, novel} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Score at 30\% params & P & PMP & P & PMP & P & PMP \\", r"\midrule"]
    for lab, d in refs:
        cells = []
        for v, k, s in cols:
            cells.append(pm(d.get((v, k, s))) if s == "P" else "")
        if lab.startswith("Dense, S0 only"):
            L.append(lab + " & \\multicolumn{2}{c}{" + pm(d[(PV, 'novel_5', 'P')]) + "} & \\multicolumn{2}{c}{" + pm(d[(PV, 'plantdoc_5', 'P')])
                     + "} & \\multicolumn{2}{c}{" + pm(d[(ES13, 'novel_5', 'P')]) + r"} \\")
        else:
            L.append(lab + " & \\multicolumn{2}{c}{" + pm(d[(PV, 'novel_5', 'P')]) + "} & \\multicolumn{2}{c}{" + pm(d[(PV, 'plantdoc_5', 'P')])
                     + "} & \\multicolumn{2}{c}{" + pm(d[(ES13, 'novel_5', 'P')]) + r"} \\")
    L.append(r"\midrule")
    for i, (lab, d) in enumerate(rows):
        L.append(lab + " & " + " & ".join(pm(d[col], 1, best[col][i]) for col in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


PAIR_COLS = [(PV, "novel_5"), (PV, "plantdoc_5"), (ES13, "novel_5")]


def table_pairs(R):
    """Paired differences (points) between choices, mean over seeds with 95% t-interval over seeds."""
    def row(label, a, b):
        cells = []
        for v, k in PAIR_COLS:
            ia, ib = (a(v) if callable(a) else a), (b(v) if callable(b) else b)
            cells.append(pd_cell(R.paired(v, ia, ib, k)))
        return label + " & " + " & ".join(cells) + r" \\"

    P = lambda c: (lambda v: R.pid(v, c))
    PMP = lambda c: f"A_{c}_PMP_30"
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Difference (a $-$ b), points & PlantVillage & PlantDoc & EuroSAT-MS \\", r"\midrule",
         r"\multicolumn{4}{@{}l}{\emph{Scores, schedule P}} \\"]
    for c in ("random", "l1", "taylor"):
        L.append(row(f"DACIS $-$ {CRIT_LABEL[c]}", P("dacis"), P(c)))
    L.append(r"\multicolumn{4}{@{}l}{\emph{Scores, schedule PMP}} \\")
    for c in ("random", "l1", "taylor"):
        L.append(row(f"DACIS $-$ {CRIT_LABEL[c]}", PMP("dacis"), PMP(c)))
    L.append(r"\multicolumn{4}{@{}l}{\emph{Schedules, DACIS}} \\")
    L.append(row("PMP $-$ P", PMP("dacis"), P("dacis")))
    L.append(row("P$\\to$M $-$ P", "B_dacis_PM_30", P("dacis")))
    L.append(row("M$\\to$P $-$ P", "B_dacis_MP_30", P("dacis")))
    L.append(row("P$\\to$G$\\to$P (no weight update) $-$ P", "B_dacis_PGP_30", P("dacis")))
    L.append(row("P$\\to$M$\\to$P$\\to$M $-$ PMP", "B_dacis_PMPM_30", PMP("dacis")))
    L.append(row("P$\\to$M ($2\\times$ episodes) $-$ P$\\to$M", "B_dacis_PM2x_30", "B_dacis_PM_30"))
    L.append(r"\multicolumn{4}{@{}l}{\emph{Episodic stage on the dense network}} \\")
    L.append(row("Dense+M $-$ Dense", "dense_m", "dense_ce"))
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_sched(R):
    rows = [("P", lambda v: R.pid(v, "dacis")), ("P$\\to$M", lambda v: "B_dacis_PM_30"),
            ("P$\\to$M ($2\\times$)", lambda v: "B_dacis_PM2x_30"), ("M$\\to$P", lambda v: "B_dacis_MP_30"),
            ("P$\\to$G$\\to$P", lambda v: "B_dacis_PGP_30"), ("P$\\to$M$\\to$P (PMP)", lambda v: "A_dacis_PMP_30"),
            ("P$\\to$M$\\to$P$\\to$M", lambda v: "B_dacis_PMPM_30")]
    L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Schedule (DACIS, 30\%) & PV 1-shot & PV 5-shot & PlantDoc 5 & EuroSAT 5 \\", r"\midrule"]
    for lab, f in rows:
        cells = [pm(R.ms(PV, f(PV), "novel_1")), pm(R.ms(PV, f(PV), "novel_5")), pm(R.ms(PV, f(PV), "plantdoc_5")),
                 pm(R.ms(ES13, f(ES13), "novel_5"))]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_budget(R):
    """Cost and accuracy by budget. The pruned architecture is the same for every score, so cost is per budget."""
    L = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Model & Params (M) & MACs (G) & CPU ms & GPU img/s & PV 5-shot & PlantDoc 5 \\", r"\midrule"]

    def line(label, vid, acc_id, cost_id=None):
        cost_id = cost_id or acc_id
        if acc_id is None or not R.has(vid, acc_id):
            return None
        mp = mparams(R, vid, cost_id)
        macs = R.eff(vid, cost_id, "macs_128")
        cpu = R.eff(vid, cost_id, "cpu_1thread_b1_ms", "median")
        gpu = R.eff(vid, cost_id, "gpu_b128_img_per_s")
        f = lambda x, fmt: "--" if x is None else fmt.format(x)
        return (f"{label} & {f(mp, '{:.2f}')} & {f(None if macs is None else macs / 1e9, '{:.3f}')} & {f(cpu, '{:.1f}')} & "
                f"{f(gpu, '{:.0f}')} & {pm(R.ms(vid, acc_id, 'novel_5'))} & {pm(R.ms(vid, acc_id, 'plantdoc_5'))}" + r" \\")

    items = [line("ResNet-18, dense", PV, "dense_ce")]
    for frac in (50, 30, 20, 10):
        cid = f"A_l1_PMP_30" if frac == 30 else f"C_l1_PMP_{frac}"
        aid = f"A_dacis_PMP_30" if frac == 30 else f"C_dacis_PMP_{frac}"
        items.append(line(f"ResNet-18, {frac}\\% (DACIS)", PV, aid, cid))
    items.append(r"\midrule")
    for arch, lab in (("mobilenet_v3_small", "MobileNetV3-Small"), ("mobilenet_v2", "MobileNetV2"), ("efficientnet_b0", "EfficientNet-B0")):
        items.append(line(lab + ", dense", PV, f"K_{arch}"))
    L += [i for i in items if i]
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_budget_scores(R):
    """5-shot PlantVillage accuracy per score at each budget (PMP schedule)."""
    L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Params kept & Random & Filter $\ell_1$ & Taylor & DACIS \\", r"\midrule"]
    for frac in (50, 30, 20, 10):
        cells = []
        for c in ("random", "l1", "taylor", "dacis"):
            iid = f"A_{c}_PMP_30" if frac == 30 else f"C_{c}_PMP_{frac}"
            cells.append(pm(R.ms(PV, iid, "novel_5")))
        L.append(f"{frac}\\% & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_recovery(R):
    L = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"PV 5-shot, schedule P, 30\% & 1 epoch & 3 epochs & 6 epochs \\", r"\midrule"]
    for c in ("random", "l1", "taylor", "dacis"):
        ids = [R.pid(PV, c), f"R_{c}_P_30_r3", f"R_{c}_P_30_r6"]
        L.append(f"{CRIT_LABEL[c]} & " + " & ".join(pm(R.ms(PV, i, "novel_5")) for i in ids) + r" \\")
    L.append(r"\midrule")
    L.append(r"Dense, no pruning & \multicolumn{3}{c}{" + pm(R.ms(PV, "dense_ce", "novel_5")) + r"} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_ablation(R):
    items = [("DACIS (full)", "A_dacis_PMP_30"),
             ("$\\mathcal{G}$ only", "A_G_PMP_30"), ("$\\mathcal{V}$ only", "A_V_PMP_30"), ("$\\mathcal{D}$ only", "A_D_PMP_30"),
             ("$\\mathcal{G}{+}\\mathcal{V}$", "A_GV_PMP_30"), ("$\\mathcal{G}{+}\\mathcal{D}$", "A_GD_PMP_30"),
             ("$\\mathcal{V}{+}\\mathcal{D}$", "A_VD_PMP_30"),
             ("no meta-gradient term", "E_dacis_PMP_30_norefine"), ("same width at every depth", "E_dacis_PMP_30_uniform"),
             ("min-max instead of ranks", "E_dacis_PMP_30_minmax")]
    L = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"DACIS variant (PMP, 30\%) & PV 5-shot & $\Delta$ vs. full \\", r"\midrule"]
    for lab, id_ in items:
        if not R.has(PV, id_):
            continue
        d = R.paired(PV, id_, "A_dacis_PMP_30", "novel_5") if id_ != "A_dacis_PMP_30" else None
        L.append(f"{lab} & {pm(R.ms(PV, id_, 'novel_5'))} & {pd_cell(d) if d else '--'}" + r" \\")
        if id_ in ("A_VD_PMP_30",):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_eurosat(R):
    rows = [("13 bands, dense (S0)", ES13, "dense_ce"), ("RGB bands, dense (S0)", ES3, "dense_ce"),
            ("13 bands, dense (S0+R+M+R)", ES13, "dense_rmr"), ("RGB bands, dense (S0+R+M+R)", ES3, "dense_rmr"),
            (r"13 bands, filter $\ell_1$, PMP 30\%", ES13, "A_l1_PMP_30"), (r"RGB bands, filter $\ell_1$, PMP 30\%", ES3, "A_l1_PMP_30"),
            ("13 bands, DACIS, PMP 30\\%", ES13, "A_dacis_PMP_30"), ("RGB bands, DACIS, PMP 30\\%", ES3, "A_dacis_PMP_30")]
    L = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"EuroSAT-MS, novel classes & 1-shot & 5-shot & 10-shot \\", r"\midrule"]
    for lab, v, i in rows:
        if R.has(v, i):
            L.append(f"{lab} & " + " & ".join(pm(R.ms(v, i, k)) for k in ("novel_1", "novel_5", "novel_10")) + r" \\")
    L.append(r"\midrule")
    for i, lab in (("dense_ce", "13 bands $-$ RGB, dense"), ("A_dacis_PMP_30", "13 bands $-$ RGB, DACIS PMP")):
        if R.has(ES13, i) and R.has(ES3, i):
            diffs = [R.runs[(ES13, s, i)]["acc"]["novel_5"]["mean"] - R.runs[(ES3, s, i)]["acc"]["novel_5"]["mean"]
                     for s in R.seeds_with(ES13, i) if (ES3, s, i) in R.runs]
            d = np.array(diffs)
            if len(d) > 1:
                half = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                L.append(f"{lab} (5-shot) & \\multicolumn{{3}}{{c}}{{${sgn(d.mean())}$ {{\\scriptsize[{sgn(d.mean() - half)},{sgn(d.mean() + half)}]}}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


LAMBDAS = [(0.3, 0.2, 0.5), (1 / 3, 1 / 3, 1 / 3), (0.5, 0.2, 0.3), (0.2, 0.5, 0.3), (0.2, 0.2, 0.6), (0.1, 0.1, 0.8),
           (0.0, 0.0, 1.0), (0.5, 0.5, 0.0)]


def table_lambda(R):
    L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"$(\lambda_G,\lambda_V,\lambda_D)$ & Val 1-shot & Val 5-shot & Novel 1-shot & Novel 5-shot \\", r"\midrule"]
    for i, lam in enumerate(LAMBDAS):
        id_ = f"S_dacis_PMP_30_lam{i}"
        if not R.has(PV, id_):
            continue
        txt = "(" + ",".join(f"{x:.2f}".rstrip("0").rstrip(".") if x not in (0.0, 1.0) else str(int(x)) for x in lam) + ")"
        L.append(f"{txt} & " + " & ".join(pm(R.ms(PV, id_, k)) for k in ("val_1", "val_5", "novel_1", "novel_5")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_mlr(R):
    if not R.tuning:
        return ""
    L = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Episodic learning rate & Val 1-shot & Val 5-shot \\", r"\midrule"]
    for k, v in R.tuning["rows"].items():
        name = "none (S0 only)" if k.startswith("S0") else k
        L.append(f"{name} & {v['1']:.1f} & {v['5']:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ---- analysis helpers -------------------------------------------------------------------------
def jaccard_matrix(R, variant=PV, crits=("random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis")):
    """Mean Jaccard overlap of the channels kept by the first prune stage of the PMP runs (budget 60%)."""
    M = np.full((len(crits), len(crits)), np.nan)
    for i, a in enumerate(crits):
        for j, b in enumerate(crits):
            vals = []
            for s in R.seeds(variant):
                ra, rb = R.runs.get((variant, s, f"A_{a}_PMP_30")), R.runs.get((variant, s, f"A_{b}_PMP_30"))
                if not ra or not rb:
                    continue
                ka = {(bi, c) for bi, ch in enumerate(ra["stages"][0]["keep"]) for c in ch}
                kb = {(bi, c) for bi, ch in enumerate(rb["stages"][0]["keep"]) for c in ch}
                vals.append(len(ka & kb) / len(ka | kb))
            if vals:
                M[i, j] = np.mean(vals)
    return M, list(crits)


# ---- figures ----------------------------------------------------------------------------------
def make_figures(R, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    os.makedirs(outdir, exist_ok=True)
    seed_cols = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d", "#e45756"]
    # (1) per-seed accuracy of each score: the ranking changes with the class partition
    panels = [(PV, "novel_5", "PlantVillage, 5-shot"), (PV, "plantdoc_5", "PlantDoc, 5-shot"), (ES13, "novel_5", "EuroSAT-MS, 5-shot")]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.0))
    order = ["random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"]
    short = {"random": "Rnd", "l1": "L1", "bn_gamma": "BN", "taylor": "Tay", "G": "G", "V": "V", "D": "D", "dacis": "DAC"}
    for ax, (var, key, title) in zip(axs, panels):
        seeds = R.seeds(var)
        for si, s in enumerate(seeds):
            ys = []
            for c in order:
                iid = f"A_{c}_PMP_30"
                r = R.runs.get((var, s, iid))
                ys.append(r["acc"][key]["mean"] if r else np.nan)
            ys = np.array(ys)
            base = R.runs.get((var, s, "dense_ce"))
            ref = base["acc"][key]["mean"] if base else 0
            if np.all(np.isnan(ys)):
                continue
            ax.plot(range(len(order)), ys - np.nanmean(ys), marker="o", ms=2.5, lw=0.9, color=seed_cols[si % len(seed_cols)],
                    label=f"split {s}")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short[c] for c in order], rotation=45)
        ax.set_title(title)
    axs[0].set_ylabel("Accuracy minus mean over scores (points)")
    axs[0].legend(frameon=False, ncol=1, loc="lower left", fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(outdir, "fig_seeds.pdf"))
    plt.close(fig)
    # (2) accuracy versus parameter budget
    panels2 = [(PV, "novel_5", "PlantVillage, 5-shot"), (PV, "plantdoc_5", "PlantDoc, 5-shot"), (ES13, "novel_5", "EuroSAT-MS, 5-shot")]
    col = {"dacis": "#c0392b", "l1": "#2c6fbb", "random": "#7f7f7f", "taylor": "#2ca02c"}
    lab = {"dacis": "DACIS", "l1": r"Filter $\ell_1$", "random": "Random", "taylor": "Taylor"}
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 1.95))
    for ax, (var, key, title) in zip(axs, panels2):
        for c in ("random", "l1", "taylor", "dacis"):
            xs, ys, es = [], [], []
            for frac in (10, 20, 30, 50):
                iid = f"A_{c}_PMP_30" if frac == 30 else f"C_{c}_PMP_{frac}"
                if R.has(var, iid):
                    m, s = R.ms(var, iid, key)
                    xs.append(frac); ys.append(m); es.append(s / np.sqrt(len(R.seeds(var))))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=2.5, lw=1.0, capsize=1.5, color=col[c], label=lab[c])
        if R.has(var, "dense_ce"):
            ax.axhline(R.ms(var, "dense_ce", key)[0], ls="--", lw=0.8, color="#333", label="Dense")
        ax.set_title(title)
        ax.set_xlabel("Parameters kept (%)")
        ax.set_xticks([10, 20, 30, 50])
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].legend(loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(outdir, "fig_budget.pdf"))
    plt.close(fig)
    # (3) resolution sweep
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    for id_, lb, c in (("dense_ce", "Dense", "#333"), ("A_l1_PMP_30", r"Filter $\ell_1$, 30%", "#2c6fbb"), ("A_dacis_PMP_30", "DACIS, 30%", "#c0392b")):
        if not R.has(PV, id_):
            continue
        res = sorted({int(k) for s in R.seeds(PV) for k in R.runs[(PV, s, id_)]["acc"].get("res_sweep_5shot", {})})
        if not res:
            continue
        ys = [np.mean([R.runs[(PV, s, id_)]["acc"]["res_sweep_5shot"][str(r)] for s in R.seeds(PV)]) for r in res]
        ax.plot(res, ys, marker="o", ms=2.5, lw=1.0, color=c, label=lb)
    ax.set_xlabel("Test resolution (px; trained at 128)")
    ax.set_ylabel("5-shot accuracy (%)")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(outdir, "fig_res.pdf"))
    plt.close(fig)
    # (4) overlap of kept channels
    M, names = jaccard_matrix(R)
    if not np.all(np.isnan(M)):
        fig, ax = plt.subplots(figsize=(3.2, 2.7))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        lb = [short[n] for n in names]
        ax.set_xticks(range(len(lb)))
        ax.set_xticklabels(lb, rotation=45, ha="right")
        ax.set_yticks(range(len(lb)))
        ax.set_yticklabels(lb)
        for i in range(len(lb)):
            for j in range(len(lb)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.tight_layout(pad=0.4)
        fig.savefig(os.path.join(outdir, "fig_overlap.pdf"))
        plt.close(fig)


# ---- numbers used in the text -------------------------------------------------------------------
def numbers(R):
    """Every number the prose quotes, keyed by name. \\num{key} in the LaTeX reads them."""
    n = {}
    for (var, seed, id_), r in R.runs.items():
        pass
    for var in (PV, ES13, ES3):
        ids = sorted({k[2] for k in R.runs if k[0] == var})
        n[f"{var}:seeds"] = str(len(R.seeds(var)))
        for id_ in ids:
            if not R.has(var, id_):
                continue
            some = R.runs[(var, R.seeds(var)[0], id_)]["acc"]
            for key in some:
                if isinstance(some[key], dict) and "mean" in some[key]:
                    ms = R.ms(var, id_, key)
                    n[f"{var}:{id_}:{key}"] = f"{ms[0]:.1f}"
                    n[f"{var}:{id_}:{key}:sd"] = f"{ms[1]:.1f}"
            pf = R.vals(var, id_, None, getter=lambda r: r["param_frac"] * 100).mean()
            n[f"{var}:{id_}:pct"] = f"{pf:.1f}"
            for fld in ("params", "macs_128", "macs_224"):
                v = R.eff(var, id_, fld)
                if v is not None:
                    n[f"{var}:{id_}:{fld}"] = f"{v / 1e6:.2f}" if fld == "params" else f"{v / 1e9:.3f}"
            for fld, sub in (("cpu_1thread_b1_ms", "median"), ("cpu_all_threads_b1_ms", "median"), ("gpu_b1_ms", "median")):
                v = R.eff(var, id_, fld, sub)
                if v is not None:
                    n[f"{var}:{id_}:{fld}"] = f"{v:.1f}"
            v = R.eff(var, id_, "gpu_b128_img_per_s")
            if v is not None:
                n[f"{var}:{id_}:gpu_ips"] = f"{v:.0f}"
    # paired differences
    def addp(tag, var, a, b, key):
        p = R.paired(var, a, b, key)
        if p:
            n[f"d:{tag}"] = sgn(p["mean"])
            n[f"d:{tag}:lo"] = sgn(p["lo"])
            n[f"d:{tag}:hi"] = sgn(p["hi"])
            n[f"d:{tag}:wins"] = str(p["wins"])
            n[f"d:{tag}:n"] = str(p["n"])
    for var, key in ((PV, "novel_5"), (PV, "plantdoc_5"), (PV, "novel_1"), (ES13, "novel_5")):
        t = f"{'pv' if var == PV else 'es'}:{key}"
        for c in ("random", "l1", "taylor", "bn_gamma", "D", "G", "V"):
            addp(f"{t}:dacis-{c}:PMP", var, "A_dacis_PMP_30", f"A_{c}_PMP_30", key)
            addp(f"{t}:dacis-{c}:P", var, R.pid(var, "dacis"), R.pid(var, c), key)
        for s in ("PM", "MP", "PGP", "PMPM", "PM2x"):
            addp(f"{t}:{s}-P", var, f"B_dacis_{s}_30", R.pid(var, "dacis"), key)
        addp(f"{t}:PMP-P", var, "A_dacis_PMP_30", R.pid(var, "dacis"), key)
        addp(f"{t}:densem-dense", var, "dense_m", "dense_ce", key)
        addp(f"{t}:dense-dacisP", var, "dense_ce", R.pid(var, "dacis"), key)
        addp(f"{t}:dense-randomP", var, "dense_ce", R.pid(var, "random"), key)
        addp(f"{t}:norefine-full", var, "E_dacis_PMP_30_norefine", "A_dacis_PMP_30", key)
    for key in ("novel_5", "novel_1", "novel_10"):
        addp(f"es:{key}:rgb-13:dense", ES3, "dense_ce", "dense_ce", key) if False else None
    if R.tuning:
        n["tune:best"] = f"{R.tuning['best_lr']:g}"
        for k, v in R.tuning["rows"].items():
            n[f"tune:{k}:1"] = f"{v['1']:.1f}"
            n[f"tune:{k}:5"] = f"{v['5']:.1f}"
    h = R.host.get("host", {})
    n["hw-gpu"] = h.get("gpu", "an unnamed GPU").replace("_", " ")
    n["hw-cpu"] = h.get("cpu", "an unnamed CPU")
    n["hw-gpumem"] = str(h.get("gpu_mem_gb", ""))
    return n


def write_numbers(R, path):
    lines = ["% generated by pmp_dacis.report.write_numbers; do not edit"]
    for k, v in numbers(R).items():
        lines.append(r"\expandafter\gdef\csname num@%s\endcsname{%s}" % (k, v))
    _write(path, "\n".join(lines) + "\n")


def claims_check(R):
    return "claims check: not yet defined for this build\n"


def markdown_summary(R):
    """Markdown tables of every run (mean ± sd over seeds), straight from the saved results."""
    def cell(var, id_, key):
        ms = R.ms(var, id_, key) if R.has(var, id_) else None
        return "--" if ms is None else f"{ms[0]:.1f} ± {ms[1]:.1f}"

    out = []
    for var, title, keys, heads in (
        (PV, "PlantVillage novel classes and PlantDoc (5-way)", ("novel_1", "novel_5", "novel_10", "plantdoc_5"),
         ("1-shot", "5-shot", "10-shot", "PlantDoc 5-shot")),
        (ES13, "EuroSAT-MS (13 bands) novel classes (5-way)", ("novel_1", "novel_5", "novel_10"), ("1-shot", "5-shot", "10-shot")),
        (ES3, "EuroSAT-MS (RGB bands only) novel classes (5-way)", ("novel_1", "novel_5", "novel_10"), ("1-shot", "5-shot", "10-shot")),
    ):
        ids = sorted({k[2] for k in R.runs if k[0] == var})
        if not ids:
            continue
        out.append(f"### {title}\n")
        out.append("| run | params kept | " + " | ".join(heads) + " |")
        out.append("|---|---|" + "---|" * len(heads))
        for id_ in ids:
            if not R.has(var, id_):
                continue
            pf = R.vals(var, id_, None, getter=lambda r: r["param_frac"] * 100).mean()
            out.append(f"| {id_} | {pf:.1f}% | " + " | ".join(cell(var, id_, k) for k in keys) + " |")
        out.append("")
    return "\n".join(out)


def build_all(out_root, gen_dir):
    """Write every generated table / figure / number used by the paper into gen_dir."""
    R = Results(out_root)
    os.makedirs(gen_dir, exist_ok=True)
    for name, fn in (("tab_scores", table_scores), ("tab_pairs", table_pairs), ("tab_sched", table_sched),
                     ("tab_budget", table_budget), ("tab_budget_scores", table_budget_scores), ("tab_recovery", table_recovery),
                     ("tab_ablation", table_ablation), ("tab_eurosat", table_eurosat), ("tab_lambda", table_lambda),
                     ("tab_mlr", table_mlr)):
        try:
            _write(os.path.join(gen_dir, name + ".tex"), fn(R))
        except Exception as e:  # a missing run must not stop the other outputs
            _write(os.path.join(gen_dir, name + ".tex"), f"% {name} could not be built: {e!r}\n")
            print("WARNING:", name, repr(e))
    make_figures(R, gen_dir)
    _write(os.path.join(gen_dir, "summary.md"), markdown_summary(R))
    return R
