"""Minimal sklearn regression training job for K8s.

Hyperparameters arrive via env vars so the SDK can fan out a grid
search as parallel Jobs, one per combination.
"""
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

alpha = float(os.environ.get("ALPHA", "1.0"))
print(f"alpha: {alpha}")

rng = np.random.default_rng(42)
X = rng.uniform(-10, 10, size=(50_000, 3))
true_coefs = np.array([1.0, -1.0, 0.5])
y = X @ true_coefs + rng.normal(0, 0.5, size=50_000)
print(f"samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"coefficients: {model.coef_.round(3).tolist()}")
print(f"intercept: {model.intercept_:.3f}")
print(f"rmse: {mean_squared_error(y_test, preds) ** 0.5:.4f}")
print(f"r2: {r2_score(y_test, preds):.4f}")

print("training complete")
