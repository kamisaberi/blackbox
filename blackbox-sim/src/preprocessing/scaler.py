import numpy as np

class FeatureScaler:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, data):
        """
data: numpy array of shape (N, 128)
        """
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)

        # Avoid division by zero in C++
        diff = self.max_vals - self.min_vals
        self.max_vals[diff == 0] += 1.0

    def transform(self, data):
        return (data - self.min_vals) / (self.max_vals - self.min_vals)

    def save_params(self, path):
        """
Exports scaler_params.txt for C++.
Format: min,max (one line per feature)
        """
        with open(path, 'w') as f:
            for i in range(len(self.min_vals)):
                f.write(f"{self.min_vals[i]:.6f},{self.max_vals[i]:.6f}\n")

        print(f"[SIM] Scaler params exported to {path}")