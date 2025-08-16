# experiments.py
# Refactored "Assumption Mismatch" experiments
# Requirements: numpy, matplotlib, scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utilities import *

RNG = np.random.default_rng(42)


# -----------------------------
# Part 1 — Wrong Assumptions → Outliers
# -----------------------------
def part1_wrong_assumptions():
    print("\n=== Part 1: Wrong Assumptions → Outliers ===")
    X = gen_gaussian(n=600, mean=np.array([0, 0]), cov=np.array([[1.0, 0.8], [0.8, 1.2]]), d=2)

    # Detector that ASSUMES wrong parameters (shifted mean and incorrect std)
    wrong_mean = np.array([2.0, -1.5])
    wrong_std  = np.array([0.5, 0.5])
    s_wrong = zscore_scores(X, assume_mean=wrong_mean, assume_std=wrong_std)
    thr_wrong = threshold_by_quantile(s_wrong, 0.95)
    out_wrong = s_wrong >= thr_wrong

    # Detector with estimated parameters from data (better assumption)
    est_mean = X.mean(axis=0)
    est_std  = X.std(axis=0) + 1e-12
    s_right = zscore_scores(X, assume_mean=est_mean, assume_std=est_std)
    thr_right = threshold_by_quantile(s_right, 0.95)
    out_right = s_right >= thr_right

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].scatter(X[:,0], X[:,1], s=10, alpha=0.6, label="data")
    ax[0].scatter(X[out_wrong,0], X[out_wrong,1], s=20, edgecolor="k", facecolor="none", label="flagged (wrong)")
    ax[0].set_title("Wrong assumptions → many false outliers")
    ax[0].legend()

    ax[1].scatter(X[:,0], X[:,1], s=10, alpha=0.6, label="data")
    ax[1].scatter(X[out_right,0], X[out_right,1], s=20, edgecolor="k", facecolor="none", label="flagged (estimated)")
    ax[1].set_title("Estimated params → fewer outliers")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

# -----------------------------
# Part 2 — Controlled Experiments → Quantifying Risks
# -----------------------------
def run_one_experiment(X_base, name, ano_ratio=0.05):
    X, y = inject_point_anomalies(X_base, ratio=ano_ratio)

    results = {}
    s1 = zscore_scores(X)  # estimates mean/std from X (practical scenario)
    results["Z-Score"] = evaluate(s1, y, name=f"{name} | Z-Score", q=0.95)

    s2 = iqr_scores(X)
    results["IQR"] = evaluate(s2, y, name=f"{name} | IQR", q=0.95)

    s3 = isolation_forest_scores(X)
    results["IsolationForest"] = evaluate(s3, y, name=f"{name} | IsolationForest", q=0.95)

    return results, {"scores": {"Z-Score": s1, "IQR": s2, "IsolationForest": s3}, "y": y}

def part2_controlled_experiments():
    print("\n=== Part 2: Controlled Experiments → Quantifying Risks ===")
    datasets = {
        "Gaussian": gen_gaussian(n=1500, d=2),
        "Exponential": gen_exponential(n=1500, lambdas=np.array([1.0, 1.5]), d=2),
        "LogNormal": gen_lognormal(n=1500, mean=0.0, sigma=0.6, d=2),
        "Bimodal": gen_bimodal(n=1500, sep=3.0, d=2),
    }

    summary = {}
    for name, X in datasets.items():
        print(f"\n--- Dataset: {name} ---")
        res, pack = run_one_experiment(X, name, ano_ratio=0.05)
        summary[name] = res

    # Plot F1 per dataset/method
    methods = ["Z-Score", "IQR", "IsolationForest"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    xs = np.arange(len(datasets))
    for m_i, m in enumerate(methods):
        vals = [summary[d][m]["f1"] for d in datasets.keys()]
        ax.bar(xs + (m_i-1)*width, vals, width, label=m)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(datasets.keys()))
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_title("Method performance across distributional shapes")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Optional ROC curves on one interesting set (e.g., Exponential)
    X = datasets["Exponential"]
    res, pack = run_one_experiment(X, "Exponential (ROC demo)", ano_ratio=0.05)
    y = pack["y"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in methods:
        s = pack["scores"][m]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{m} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1], [0,1], linestyle="--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC on Exponential data")
    ax.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Part 3 — Structured Workflow
# -----------------------------
def check_distribution_shape(X):
    """Simple diagnostics: per-feature skewness and kurtosis (excess)."""
    # unbiased-ish sample central moments
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    Z = (X - mu) / std
    skew = (np.mean(Z**3, axis=0))
    kurt = (np.mean(Z**4, axis=0) - 3.0)  # excess kurtosis
    return skew, kurt

def run_workflow(Xgen_fn, label="Dataset", detector="auto", n=1200):
    # 1) Generate data
    X = Xgen_fn(n=n)

    # 2) Quick distribution diagnostics
    skew, kurt = check_distribution_shape(X)

    # 3) Inject anomalies for evaluation
    X2, y = inject_point_anomalies(X, ratio=0.05)

    # 4) Choose detector
    if detector == "z":
        scores = zscore_scores(X2)
        det_name = "Z-Score"
    elif detector == "iqr":
        scores = iqr_scores(X2)
        det_name = "IQR"
    else:
        scores = isolation_forest_scores(X2)
        det_name = "IsolationForest"

    # 5) Evaluate
    metrics = evaluate(scores, y, name=f"{label} | {det_name}", q=0.95)

    # 6) Visualize
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].scatter(X[:,0], X[:,1], s=8, alpha=0.6)
    ax[0].set_title(f"{label}: raw data")
    thr = threshold_by_quantile(scores, 0.95)
    flagged = scores >= thr
    ax[1].scatter(X2[:,0], X2[:,1], s=8, alpha=0.6, label="points")
    ax[1].scatter(X2[flagged,0], X2[flagged,1], s=20, edgecolor="k", facecolor="none", label="flagged")
    ax[1].set_title(f"{label}: {det_name} flagged")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # 7) Print diagnostics
    print(f"{label} — feature skewness (mean over dims): {np.mean(np.abs(skew)):.2f}")
    print(f"{label} — feature excess kurtosis (mean over dims): {np.mean(np.abs(kurt)):.2f}")
    return metrics

def draw_flowchart():
    """Simple flowchart with matplotlib annotations (no external files)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    def box(xy, text):
        ax.add_patch(plt.Rectangle((xy[0], xy[1]), 0.6, 0.12, fill=False))
        ax.text(xy[0]+0.3, xy[1]+0.06, text, ha="center", va="center")

    def arrow(p1, p2):
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->"))

    box((0.2, 0.8), "Generate Dataset\n(Normal / Exponential / etc.)")
    box((0.2, 0.6), "Check Assumptions\n(skew/kurtosis)")
    box((0.2, 0.4), "Choose Detector\n(Z, IQR, IsolationForest)")
    box((0.2, 0.2), "Evaluate\n(F1 / ROC-AUC)")
    box((0.2, 0.0), "Decide & Document")

    arrow((0.5, 0.8), (0.5, 0.72))
    arrow((0.5, 0.6), (0.5, 0.52))
    arrow((0.5, 0.4), (0.5, 0.32))
    arrow((0.5, 0.2), (0.5, 0.12))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.set_title("Workflow: Test assumptions before deploying an algorithm")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Part 1
    part1_wrong_assumptions()

    # Part 2
    part2_controlled_experiments()

    # Part 3: workflow runs
    print("\n=== Part 3: Structured Workflow ===")
    _ = run_workflow(lambda n: gen_gaussian(n=n, d=2), label="Gaussian", detector="z", n=1200)
    _ = run_workflow(lambda n: gen_exponential(n=n, lambdas=np.array([1.0, 1.5]), d=2), label="Exponential", detector="auto", n=1200)

    draw_flowchart()
