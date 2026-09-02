"""Ensemble mean, spread, quantiles, and threshold exceedance features."""

def ensemble_summary(member_values):
    import numpy as np
    values = np.asarray(member_values, dtype=float)
    return {"mean": float(values.mean()), "spread": float(values.std()),
            "minimum": float(values.min()), "maximum": float(values.max())}
