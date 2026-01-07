"""
Deep Learning-Based Preprocessing for Raman Spectra

This module contains neural network-based preprocessing methods for
unified denoising and baseline correction.

Methods:
- Convolutional Autoencoder (CDAE): Data-driven noise and baseline removal

Note: This module requires PyTorch. Training data is needed before use.
"""

import numpy as np
from typing import Optional, Tuple, Literal
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. CDAE methods will not work.")

try:
    import ramanspy as rp

    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False

try:
    from ..utils import create_logs
except ImportError:
    # Fallback logging function
    def create_logs(log_id, source, message, status="info"):
        print(f"[{status.upper()}] {source}: {message}")


if TORCH_AVAILABLE:

    class Conv1DAutoencoder(nn.Module):
        """
        1D Convolutional Autoencoder for Raman Spectra

        Architecture:
        - Encoder: 3 Conv1D layers with ReLU, reducing dimensionality
        - Latent space: Compressed representation (16-32 dims typical)
        - Decoder: 3 ConvTranspose1D layers, reconstructing spectrum

        Purpose:
        - Remove noise while preserving narrow Raman peaks
        - Optionally remove baseline via multi-task learning
        - Self-supervised or supervised training on clean/noisy pairs
        """

        def __init__(
            self,
            input_size: int,
            latent_dim: int = 32,
            kernel_sizes: Tuple[int, int, int] = (7, 11, 15),
        ):
            """
            Initialize autoencoder architecture.

            Args:
                input_size: Length of input spectrum
                latent_dim: Dimensionality of latent space (16-32 typical)
                kernel_sizes: Kernel sizes for conv layers (odd numbers)
            """
            super().__init__()

            self.input_size = input_size
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(
                    1, 32, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2
                ),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(
                    32, 64, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2
                ),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(
                    64, 128, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

            # Latent layer
            self.encoder_fc = nn.Linear(128, latent_dim)
            self.decoder_fc = nn.Linear(latent_dim, 128)

            # Decoder
            decoder_size = input_size // 4  # After 2 pooling layers
            self.decoder_size = decoder_size

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(
                    128,
                    64,
                    kernel_size=kernel_sizes[2],
                    stride=2,
                    padding=kernel_sizes[2] // 2,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.ConvTranspose1d(
                    64,
                    32,
                    kernel_size=kernel_sizes[1],
                    stride=2,
                    padding=kernel_sizes[1] // 2,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    32, 1, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2
                ),
            )

        def encode(self, x):
            """Encode spectrum to latent representation."""
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
            return self.encoder_fc(x)

        def decode(self, z):
            """Decode latent representation to spectrum."""
            x = self.decoder_fc(z)
            x = x.view(x.size(0), 128, 1)
            # Upsample to decoder_size
            x = x.repeat(1, 1, self.decoder_size)
            return self.decoder(x)

        def forward(self, x):
            """Full forward pass: encode -> decode."""
            z = self.encode(x)
            return self.decode(z)

    class ConvolutionalAutoencoder:
        """
            Convolutional Autoencoder (CDAE) for Raman Preprocessing

        Purpose:
        - Learn data-driven mapping that removes noise and baseline
        - Preserve narrow Raman peaks through specialized architecture
        - Unified preprocessing in single learned transformation

        Training Objectives:
        1. Reconstruction MSE: L_rec = ||x_clean - x_hat||²
        2. Total Variation penalty (optional): L_TV = λ Σ|x_hat[j] - x_hat[j-1]|
        3. Baseline head (multi-task, optional): L_base = ||b - B(E(x))||²

        Total loss: L = L_rec + α*L_TV + β*L_base

        Training Data Requirements:
        - Clean spectra (or synthesized from library)
        - Noise/baseline synthesis: AR(1), 1/f noise, polynomial baselines
        - Typical training: 500-5000 spectra, 50-100 epochs

        Usage Pattern:
        1. Initialize with spectrum length
        2. Train on clean/noisy pairs: model.train_model(clean, noisy)
        3. Apply to new spectra: model.transform(noisy_spectra)

        References:
        - MDPI Sensors (2024) - CDAE for Raman denoising
        - SPIE (2024) - Multi-task deep learning for preprocessing
        - Nature (2024) - Self-supervised Raman preprocessing
        """

    def __init__(
        self,
        input_size: Optional[int] = None,
        latent_dim: int = 32,
        kernel_sizes: Tuple[int, int, int] = (7, 11, 15),
        tv_weight: float = 0.01,
        device: Optional[str] = None,
    ):
        """
        Initialize CDAE preprocessor.

        Args:
            input_size: Length of input spectra (auto-detected if None)
            latent_dim: Latent space dimensionality (16-32 typical)
            kernel_sizes: Conv kernel sizes for each layer
            tv_weight: Weight for total variation regularization
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ConvolutionalAutoencoder")

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.kernel_sizes = kernel_sizes
        self.tv_weight = tv_weight

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.is_trained = False

        create_logs(
            "cdae_init",
            "ConvolutionalAutoencoder",
            f"Initialized CDAE with latent_dim={latent_dim}, device={self.device}",
            status="info",
        )

    def _build_model(self, input_size: int):
        """Build model architecture."""
        self.model = Conv1DAutoencoder(
            input_size=input_size,
            latent_dim=self.latent_dim,
            kernel_sizes=self.kernel_sizes,
        ).to(self.device)

        self.input_size = input_size

    def train_model(
        self,
        clean_spectra: np.ndarray,
        noisy_spectra: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
    ):
        """
        Train autoencoder on clean/noisy spectrum pairs.

        Args:
            clean_spectra: 2D array (n_samples, n_features) of clean spectra
            noisy_spectra: 2D array of noisy/degraded spectra (same shape)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Adam optimizer learning rate
            validation_split: Fraction of data for validation
        """
        if clean_spectra.shape != noisy_spectra.shape:
            raise ValueError("Clean and noisy spectra must have same shape")

        # Build model if not already built
        if self.model is None:
            self._build_model(clean_spectra.shape[1])

        # Convert to torch tensors
        clean_tensor = torch.FloatTensor(clean_spectra).unsqueeze(1)  # Add channel dim
        noisy_tensor = torch.FloatTensor(noisy_spectra).unsqueeze(1)

        # Split train/validation
        n_samples = len(clean_tensor)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = torch.randperm(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        train_dataset = TensorDataset(noisy_tensor[train_idx], clean_tensor[train_idx])
        val_dataset = TensorDataset(noisy_tensor[val_idx], clean_tensor[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        mse_loss = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for noisy_batch, clean_batch in train_loader:
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                reconstructed = self.model(noisy_batch)

                # Reconstruction loss
                loss_rec = mse_loss(reconstructed, clean_batch)

                # Total variation regularization
                loss_tv = torch.mean(
                    torch.abs(reconstructed[:, :, 1:] - reconstructed[:, :, :-1])
                )

                # Total loss
                loss = loss_rec + self.tv_weight * loss_tv

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for noisy_batch, clean_batch in val_loader:
                        noisy_batch = noisy_batch.to(self.device)
                        clean_batch = clean_batch.to(self.device)
                        reconstructed = self.model(noisy_batch)
                        val_loss += mse_loss(reconstructed, clean_batch).item()

                create_logs(
                    "cdae_training",
                    "ConvolutionalAutoencoder",
                    f"Epoch {epoch}/{epochs}: train_loss={train_loss/len(train_loader):.6f}, "
                    + f"val_loss={val_loss/len(val_loader):.6f}",
                    status="info",
                )
                self.model.train()

        self.is_trained = True
        create_logs(
            "cdae_training_complete",
            "ConvolutionalAutoencoder",
            f"Training complete after {epochs} epochs",
            status="info",
        )

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply trained autoencoder to denoise/remove baseline.

        Args:
            spectra: 1D or 2D array of spectra

        Returns:
            Denoised/baseline-corrected spectra
        """
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before transform(). Call train_model() first."
            )

        self.model.eval()

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        # Convert to tensor
        spectra_tensor = torch.FloatTensor(spectra).unsqueeze(1).to(self.device)

        # Apply model
        with torch.no_grad():
            denoised_tensor = self.model(spectra_tensor)

        # Convert back to numpy
        denoised = denoised_tensor.squeeze(1).cpu().numpy()

        if squeeze:
            denoised = denoised.squeeze(0)

        return denoised

    def __call__(self, spectra: np.ndarray) -> np.ndarray:
        """Make callable for pipeline compatibility."""
        return self.transform(spectra)

    def apply(self, spectra: "rp.SpectralContainer") -> "rp.SpectralContainer":
        """Apply to ramanspy SpectralContainer."""
        if not RAMANSPY_AVAILABLE:
            raise ImportError("ramanspy required for SpectralContainer operations")

        data = spectra.spectral_data
        processed_data = self.transform(data)

        return rp.SpectralContainer(processed_data, spectra.spectral_axis)

    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            create_logs(
                "cdae_save_warning",
                "ConvolutionalAutoencoder",
                "Model not trained, saving untrained weights",
                status="warning",
            )

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "latent_dim": self.latent_dim,
                "kernel_sizes": self.kernel_sizes,
                "is_trained": self.is_trained,
            },
            filepath,
        )

        create_logs(
            "cdae_save",
            "ConvolutionalAutoencoder",
            f"Model saved to {filepath}",
            status="info",
        )

    def load_model(self, filepath: str):
        """Load trained model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.input_size = checkpoint["input_size"]
        self.latent_dim = checkpoint["latent_dim"]
        self.kernel_sizes = checkpoint["kernel_sizes"]
        self.is_trained = checkpoint["is_trained"]

        self._build_model(self.input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        create_logs(
            "cdae_load",
            "ConvolutionalAutoencoder",
            f"Model loaded from {filepath}",
            status="info",
        )

else:
    # Stub class when PyTorch is not available
    class ConvolutionalAutoencoder:
        """Stub class - PyTorch not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for ConvolutionalAutoencoder. "
                "Install with: pip install torch"
            )
