import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class F1TelemetryDataset(Dataset):
    """
    Transforms uniform 50Hz time-series dataframes into sequential 3D tensors 
    optimized for recurrent neural network input structures.
    """
    def __init__(self, dataframe, feature_cols, target_col, sequence_length=50):
        self.X = torch.tensor(dataframe[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target_col].values, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        # Prevent boundary out-of-index slicing during sequence windowing
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        # Extract a continuous slice of historical context up to time t
        x_sequence = self.X[idx : idx + self.sequence_length]
        # Map sequence directly to the future target parameter at time t
        y_target = self.y[idx + self.sequence_length]
        return x_sequence, y_target


class HybridVirtualSensor(nn.Module):
    """
    Physics-Informed Deep Learning Core combining spatial feature extraction 
    via 1D Temporal Convolutions and macro thermal tracking via Bi-LSTMs.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, sequence_length: int = 50, dropout_rate: float = 0.2):
        super(HybridVirtualSensor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Spatial Feature Extraction Module: Evaluates relationships across parallel sensor fields
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Temporal State Extractor: Evaluates bidirectional cumulative heat propagation curves over time
        self.bi_lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0.0
        )
        
        # Multilayer Regressor Head: Projects deep representation down to a continuous temperature scalar
        self.regressor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), # Hidden dim doubled to account for concatenated Forward + Backward passes
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Input Shape Profile: [Batch Size, Sequence Length, Input Channels]
        """
        # PyTorch Conv1d expects shape layout: [Batch Size, Input Channels, Sequence Length]
        x_transposed = x.transpose(1, 2)
        cnn_features = self.temporal_cnn(x_transposed)
        
        # Revert tensor dimensionality shape configuration back for Recurrent layer digestion
        lstm_input = cnn_features.transpose(1, 2)
        lstm_out, (hn, cn) = self.bi_lstm(lstm_input)
        
        # Isolate the final time step hidden representation matrix across the sequence block
        final_timestep_features = lstm_out[:, -1, :]
        
        # Generate the virtual estimation scalar value
        predicted_temperature = self.regressor_head(final_timestep_features)
        return predicted_temperature

    def save_checkpoint(self, path: str, scalar_metadata=None):
        """Encapsulates model weights along with contextual hyperparameters for deployment validation."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        checkpoint = {
            'state_dict': self.state_dict(),
            'input_dim': self.temporal_cnn[0].in_channels,
            'hidden_dim': self.hidden_dim,
            'sequence_length': self.sequence_length,
            'scalar_metadata': scalar_metadata
        }
        torch.save(checkpoint, path)
        print(f"[SUCCESS] Saved Virtual Sensor configuration payload to: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu"):
        """Dynamically reconstructs full neural network mapping profiles from disk storage targets."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            sequence_length=checkpoint['sequence_length']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model, checkpoint['scalar_metadata']