"""Calibrate final probabilities on a time-separated validation set."""

def calibrate(model, features, labels, method="isotonic"):
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(model, method=method, cv="prefit").fit(features, labels)
