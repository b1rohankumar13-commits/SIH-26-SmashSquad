"""Classification metrics for rare bust events.

Two families:
- Threshold-free ranking/probabilistic scores (PR-AUC, ROC-AUC, Brier).
- Operating-point (confusion-matrix) scores the meteorological community uses for
  rare events: POD/hit-rate, FAR, CSI, and the base-rate-independent TSS (Peirce)
  and SEDI. These let a domain expert compare us to known extreme-rain skill.
"""

import numpy as np


def classification_metrics(labels, probabilities, threshold=0.5):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    return {"pr_auc": average_precision_score(labels, probabilities),
            "roc_auc": roc_auc_score(labels, probabilities),
            "brier": brier_score_loss(labels, probabilities)}


def contingency(labels, probabilities, threshold):
    """2x2 contingency counts at a probability threshold (labels/probs are 1-D, finite)."""
    labels = np.asarray(labels).astype(bool)
    predicted = np.asarray(probabilities) >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    return tp, fp, fn, tn


def skill_scores(tp, fp, fn, tn):
    """POD/FAR/CSI/bias/precision + base-rate-independent TSS (Peirce) and SEDI."""
    pod = tp / (tp + fn) if (tp + fn) else float("nan")          # hit rate / recall
    far = fp / (tp + fp) if (tp + fp) else float("nan")          # false-alarm ratio
    pofd = fp / (fp + tn) if (fp + tn) else float("nan")         # false-alarm rate
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")  # critical success index
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    bias = (tp + fp) / (tp + fn) if (tp + fn) else float("nan")
    tss = pod - pofd                                             # Peirce / true skill statistic
    # Symmetric Extremal Dependence Index (base-rate independent, for rare events).
    h, f = pod, pofd
    if 0 < h < 1 and 0 < f < 1:
        sedi = ((np.log(f) - np.log(h) - np.log(1 - f) + np.log(1 - h))
                / (np.log(f) + np.log(h) + np.log(1 - f) + np.log(1 - h)))
    else:
        sedi = float("nan")
    return {"POD": pod, "FAR": far, "POFD": pofd, "CSI": csi, "precision": precision,
            "frequency_bias": bias, "TSS": tss, "SEDI": sedi,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def operating_point_table(labels, probabilities, thresholds=None):
    """Skill scores across thresholds plus the CSI- and TSS-optimal operating points."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if thresholds is None:
        thresholds = np.unique(np.quantile(probabilities, np.linspace(0.5, 0.9995, 40)))
    rows = []
    for t in thresholds:
        rows.append({"threshold": float(t), **skill_scores(*contingency(labels, probabilities, t))})
    best_csi = max(rows, key=lambda r: (r["CSI"] if np.isfinite(r["CSI"]) else -1))
    best_tss = max(rows, key=lambda r: (r["TSS"] if np.isfinite(r["TSS"]) else -1))
    return {"rows": rows, "best_csi": best_csi, "best_tss": best_tss,
            "base_rate": float(np.mean(labels))}
