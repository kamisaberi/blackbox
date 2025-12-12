import torch
import torch.nn as nn

class LogAutoencoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, bottleneck_dim=32):
        super(LogAutoencoder, self).__init__()

        # Encoder: Compress the 128-float vector into a tiny representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )

        # Decoder: Try to reconstruct the original vector
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            # We use Sigmoid because FeatureScaler normalizes inputs to [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded