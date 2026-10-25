"""Preprocessing step — must complete before any training job starts.

Placeholder for now: validates the synthetic data pipeline without
persisting artifacts (no PVC/S3 writes yet).
"""
import numpy as np

print("loading raw data...")
rng = np.random.default_rng(42)
X = rng.uniform(-10, 10, size=(50_000, 3))
print(f"loaded {len(X)} samples, {X.shape[1]} features")

print("validating features...")
means = X.mean(axis=0).round(3).tolist()
stds = X.std(axis=0).round(3).tolist()
print(f"feature means: {means}")
print(f"feature stds: {stds}")
assert not np.isnan(X).any(), "NaNs detected in features"

print("preprocessing complete")
