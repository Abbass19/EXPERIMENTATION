import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score



RNG = np.random.default_rng(42)


def gen_gaussian(n=1000, mean=None, cov=None, d=2):
    if mean is None: mean = np.zeros(d)
    if cov is None:  cov  = np.eye(d)
    return RNG.multivariate_normal(mean, cov, size=n)

def gen_exponential(n=1000, lambdas=None, d=2):
    if lambdas is None: lambdas = np.ones(d)
    X = np.column_stack([RNG.exponential(scale=1/l, size=n) for l in lambdas])
    return X

def gen_lognormal(n=1000, mean=0.0, sigma=0.5, d=2):
    return RNG.lognormal(mean=mean, sigma=sigma, size=(n, d))

def gen_bimodal(n=1000, sep=3.0, d=2):
    n1 = n // 2
    n2 = n - n1
    c1 = gen_gaussian(n1, mean=np.zeros(d), cov=np.eye(d)*0.5, d=d)
    c2 = gen_gaussian(n2, mean=np.ones(d)*sep, cov=np.eye(d)*0.5, d=d)
    return np.vstack([c1, c2])

def inject_point_anomalies(X, ratio=0.05, scale=8.0):
    """Inject point anomalies by adding large noise to a small subset."""
    n = len(X)
    k = max(1, int(n * ratio))
    idx = RNG.choice(n, size=k, replace=False)
    X_ano = X.copy()
    X_ano[idx] = X_ano[idx] + RNG.normal(0, scale, size=X_ano[idx].shape)
    y = np.zeros(n, dtype=int)
    y[idx] = 1  # anomalies
    return X_ano, y

# -----------------------------
# Utilities: detectors
# -----------------------------
def zscore_scores(X, assume_mean=None, assume_std=None):
    """Return a score = max|z| per row. Assumes independent features."""
    if assume_mean is None: assume_mean = X.mean(axis=0)
    if assume_std  is None: assume_std  = X.std(axis=0) + 1e-12
    Z = (X - assume_mean) / assume_std
    return np.max(np.abs(Z), axis=1)

def iqr_scores(X):
    """Tukey fences distance (how far past 1.5*IQR) aggregated per row."""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = (Q3 - Q1) + 1e-12
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    below = (lower - X) / IQR
    above = (X - upper) / IQR
    # Only count positive violations
    violation = np.maximum(0, np.maximum(below, above))
    return np.max(violation, axis=1)

def isolation_forest_scores(X):
    iso = IsolationForest(random_state=42, contamination="auto")
    iso.fit(X)
    # Higher score = more normal; invert so higher = more anomalous like others
    s = -iso.score_samples(X)
    return s

def threshold_by_quantile(scores, q=0.95):
    thr = np.quantile(scores, q)
    return thr

def evaluate(scores, y, name="method", q=0.95):
    thr = threshold_by_quantile(scores, q)
    yhat = (scores >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    roc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else np.nan
    print(f"[{name}]  q={q:.2f}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  ROC-AUC={roc:.3f}")
    return {"precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}