"""Classification metrics for rare bust events."""

def classification_metrics(labels, probabilities, threshold=0.5):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    return {"pr_auc": average_precision_score(labels, probabilities),
            "roc_auc": roc_auc_score(labels, probabilities),
            "brier": brier_score_loss(labels, probabilities)}
