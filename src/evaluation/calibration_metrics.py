"""Reliability and probability-calibration diagnostics."""

def brier_score(labels, probabilities):
    import numpy as np
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))
