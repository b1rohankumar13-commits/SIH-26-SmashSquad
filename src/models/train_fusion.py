"""Fuse XGBoost and ConvLSTM probabilities using logistic regression."""

def train_fusion(xgb_probability, convlstm_probability, labels):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    inputs = np.column_stack([xgb_probability, convlstm_probability])
    return LogisticRegression().fit(inputs, labels)
