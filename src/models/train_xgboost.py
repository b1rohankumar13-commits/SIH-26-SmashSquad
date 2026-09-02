"""Train the tabular bust-probability model."""

def train_xgboost(features, labels, params):
    from xgboost import XGBClassifier
    model = XGBClassifier(**params)
    return model.fit(features, labels)
