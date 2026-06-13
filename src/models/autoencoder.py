import os
import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    """
    Deep Autoencoder designed to capture multi-channel residual distributions 
    and identify system degradation via Reconstruction Loss Spikes.
    """
    def __init__(self, input_dim: int):
        super(AnomalyAutoencoder, self).__init__()
        
        # Bottleneck compression layer: Forces network to learn structural dependencies
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 4),  # Bottleneck layer
            nn.ReLU()
        )
        
        # Reconstruction layer: Restores compressed vectors back to their original dimensions
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        """
        Input Shape: [Batch Size, Input Dimensions]
        """
        latent_bottleneck = self.encoder(x)
        reconstructed_residuals = self.decoder(latent_bottleneck)
        return reconstructed_residuals

    def calculate_reconstruction_loss(self, x):
        """
        Evaluates row-by-row Mean Squared Error to isolate outlying data frames.
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Compute element-wise squared error across features
            squared_errors = torch.pow(x - reconstructed, 2)
            # Average across the feature dimensions to get per-sample loss
            sample_losses = torch.mean(squared_errors, dim=1)
        return sample_losses, reconstructed

    def save_model(self, path: str, threshold_metadata: float = None):
        """Saves weights alongside computed anomaly alerting thresholds."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        payload = {
            'state_dict': self.state_dict(),
            'input_dim': self.encoder[0].in_features,
            'alert_threshold': threshold_metadata
        }
        torch.save(payload, path)
        print(f"[SUCCESS] Saved Isolation Engine payload to: {path}")

    @classmethod
    def load_model(cls, path: str, device: str = "cpu"):
        """Loads and instantiates full autoencoder configuration parameters from disk."""
        payload = torch.load(path, map_location=device)
        model = cls(input_dim=payload['input_dim'])
        model.load_state_dict(payload['state_dict'])
        model.to(device)
        return model, payload['alert_threshold']