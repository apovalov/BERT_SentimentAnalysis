from typing import List

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


def evaluate(model, embeddings, labels, cv: int = 5) -> List[float]:
    """Evaluate model on embeddings and labels"""
    # Cross-validation
    kf = KFold(n_splits=cv, shuffle=False)
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    losses = []

    for train_index, test_index in kf.split(embeddings):
        # Split data
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict_proba(X_test)
        loss = log_loss(y_test, y_pred)
        losses.append(loss)

    return losses
