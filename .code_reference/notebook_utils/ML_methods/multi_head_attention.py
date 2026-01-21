import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import pickle
import math
import time
import gc
from collections import Counter

class SpectralDataPreprocessor:
    """
    Advanced data preprocessing for Raman spectroscopy data to fix critical data issues.
    """
    def __init__(self, normalization_method='robust', clip_outliers=True):
        self.normalization_method = normalization_method
        self.clip_outliers = clip_outliers
        self.scaler = None
        self.clip_min = None
        self.clip_max = None
        
    def fit_transform(self, X_train, y_train=None):
        """Fit preprocessing parameters and transform training data"""
        print(f"🔧 Preprocessing spectral data...")
        print(f"   Original shape: {X_train.shape}")
        print(f"   Original range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        
        X_processed = X_train.copy()
        
        # 1. Clip extreme outliers that can destroy normalization
        if self.clip_outliers:
            q1, q99 = np.percentile(X_processed, [1, 99])
            self.clip_min, self.clip_max = q1, q99
            X_processed = np.clip(X_processed, self.clip_min, self.clip_max)
            print(f"   Clipped to [{self.clip_min:.3f}, {self.clip_max:.3f}] (1-99 percentiles)")
        
        # 2. Apply normalization
        if self.normalization_method == 'robust':
            self.scaler = RobustScaler()
        elif self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        X_processed = self.scaler.fit_transform(X_processed)
        
        print(f"   After {self.normalization_method} normalization: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
        print(f"   Mean: {X_processed.mean():.3f}, Std: {X_processed.std():.3f}")
        
        return X_processed
    
    def transform(self, X):
        """Transform new data using fitted parameters"""
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        X_processed = X.copy()
        
        if self.clip_outliers:
            X_processed = np.clip(X_processed, self.clip_min, self.clip_max)
        
        X_processed = self.scaler.transform(X_processed)
        
        return X_processed

class PositionalEncoding(nn.Module):
    """Positional encoding for spectral data to maintain wavelength relationships."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EfficientMultiHeadAttention(nn.Module):
    """Simplified multi-head attention for better learning on small datasets."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super(EfficientMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.output(attended)
        
        # Residual connection and layer normalization
        return self.layer_norm(x + output)

class MCEBlock(nn.Module):
    """Simplified Multi-Component Encoder Block for small datasets."""
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = None, dropout: float = 0.2):
        super(MCEBlock, self).__init__()
        if d_ff is None:
            d_ff = 2 * d_model  # Reduced from 4x to 2x
        
        self.attention = EfficientMultiHeadAttention(d_model, n_heads, dropout)
        
        # Simplified feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU instead of ReLU for better gradients
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention
        attended = self.attention(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(attended)
        return self.layer_norm(attended + ff_output)

class SimplifiedSpectralTransformer(nn.Module):
    """
    Simplified Spectral Transformer designed for small datasets with class imbalance.
    
    Key improvements:
    - Much smaller model (16D instead of 64D)
    - Fewer layers to prevent overfitting
    - Better regularization
    - Proper initialization
    """
    
    def __init__(self, input_size: int, n_classes: int, 
                 d_model: int = 16, n_heads: int = 4, n_layers: int = 2, 
                 dropout: float = 0.3, max_seq_len: int = 2048):
        super(SimplifiedSpectralTransformer, self).__init__()
        
        self.input_size = input_size
        self.n_classes = n_classes
        self.d_model = d_model
        
        # Input projection to model dimension
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Simplified encoder blocks
        self.encoders = nn.ModuleList([
            MCEBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) 
            for _ in range(n_layers)
        ])
        
        # Global attention pooling instead of average pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head with proper regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # Initialize weights properly
        self._init_weights()
        
        print(f"SimplifiedSpectralTransformer Architecture:")
        print(f"  Input: {input_size} wavelength points")
        print(f"  Model dimension: {d_model} (simplified)")
        print(f"  Encoder layers: {n_layers}")
        print(f"  Attention heads: {n_heads}")
        print(f"  Dropout rate: {dropout}")
        print(f"  Output classes: {n_classes}")
        
    def _init_weights(self):
        """Proper weight initialization for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        # Input shape: (batch_size, input_size)
        batch_size, seq_len = x.size()
        
        # Reshape for attention: (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Project to model dimension: (batch_size, seq_len, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply encoder blocks
        for encoder in self.encoders:
            x = encoder(x)
        
        # Attention-based global pooling
        attention_weights = self.attention_pool(x)  # (batch_size, seq_len, 1)
        x = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x

class SpectralTransformerModel:
    """
    Improved wrapper class with proper data preprocessing and class balancing.
    """
    
    def __init__(self, data_split: Dict[str, Any], 
                 d_model: int = 16, n_heads: int = 4, n_layers: int = 2,
                 lr: float = 0.001, epochs: int = 100, batch_size: int = 32, 
                 dropout: float = 0.3, device: str = None, 
                 normalization_method: str = 'robust',
                 use_class_weights: bool = True,
                 use_weighted_sampling: bool = True, 
                 **kwargs):
        """
        Initialize the improved Spectral Transformer model.
        
        Args:
            data_split (dict): Data dictionary from RamanDataPreparer.prepare_data()
            d_model (int): Model dimension (default 16, much smaller)
            n_heads (int): Number of attention heads (default 4)
            n_layers (int): Number of encoder layers (default 2)
            lr (float): Learning rate (default 0.001)
            epochs (int): Training epochs (default 100)
            batch_size (int): Batch size (default 32)
            dropout (float): Dropout rate (default 0.3, higher for regularization)
            device (str): Device to use ('cuda' or 'cpu')
            normalization_method (str): 'robust', 'standard', or 'minmax'
            use_class_weights (bool): Use class weights in loss function
            use_weighted_sampling (bool): Use weighted sampling for class balance
        """
        self.data_split_raw = data_split.copy()  # Keep original data
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_class_weights = use_class_weights
        self.use_weighted_sampling = use_weighted_sampling
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check required keys
        required_keys = ['X_train', 'y_train', 'unified_wavelengths']
        missing_keys = [key for key in required_keys if key not in data_split]
        if missing_keys:
            raise ValueError(f"Missing keys in data_split: {missing_keys}")
        
        # Analyze original data
        print(f"\n=== Data Analysis ===")
        print(f"Original data shape: {data_split['X_train'].shape}")
        print(f"Data range: [{data_split['X_train'].min():.3f}, {data_split['X_train'].max():.3f}]")
        
        # Analyze class distribution
        class_counts = Counter(data_split['y_train'])
        print(f"Class distribution: {dict(class_counts)}")
        
        # Check for class imbalance
        class_ratios = [count / len(data_split['y_train']) for count in class_counts.values()]
        imbalance_ratio = max(class_ratios) / min(class_ratios)
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 1.5:
            print(f"⚠️  Significant class imbalance detected!")
        
        # Apply data preprocessing
        self.preprocessor = SpectralDataPreprocessor(normalization_method=normalization_method)
        X_train_processed = self.preprocessor.fit_transform(data_split['X_train'])
        X_test_processed = self.preprocessor.transform(data_split['X_test']) if len(data_split['X_test']) > 0 else np.array([])
        
        # Create processed data split
        self.data_split = {
            'X_train': X_train_processed,
            'y_train': data_split['y_train'],
            'X_test': X_test_processed,
            'y_test': data_split['y_test'],
            'unified_wavelengths': data_split['unified_wavelengths']
        }
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(self.data_split['y_train'])
        
        # Infer dimensions
        input_size = self.data_split['X_train'].shape[1]
        n_classes = len(np.unique(self.y_train_encoded))
        
        # Compute class weights for balanced training
        if self.use_class_weights:
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(self.y_train_encoded), 
                y=self.y_train_encoded
            )
            self.class_weights = torch.FloatTensor(class_weights).to(self.device)
            print(f"Class weights: {dict(zip(self.label_encoder.classes_, class_weights))}")
        else:
            self.class_weights = None
        
        # Initialize simplified model
        self.model = SimplifiedSpectralTransformer(
            input_size=input_size,
            n_classes=n_classes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            max_seq_len=input_size
        ).to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        
        # Loss function with class weights
        if self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n=== Model Initialization ===")
        print(f"Dataset: {len(self.data_split['X_train'])} train, {len(self.data_split['X_test'])} test samples")
        print(f"Input size: {input_size} wavelengths, Classes: {n_classes}")
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Device: {self.device}")
        print(f"Optimizer: AdamW (lr={self.lr}, weight_decay=0.01)")
        print(f"Batch size: {self.batch_size}")
        print(f"Class balancing: {'Enabled' if self.use_class_weights else 'Disabled'}")
        print(f"Weighted sampling: {'Enabled' if self.use_weighted_sampling else 'Disabled'}")
        print("=" * 50)
    
    def _create_balanced_dataloader(self):
        """Create a dataloader with balanced sampling to address class imbalance."""
        X_train = torch.tensor(self.data_split['X_train'], dtype=torch.float32)
        y_train = torch.tensor(self.y_train_encoded, dtype=torch.long)
        
        if self.use_weighted_sampling:
            # Compute sample weights for balanced sampling
            class_counts = torch.bincount(y_train)
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[y_train]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
            print(f"✅ Using weighted sampling for class balance")
        else:
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        return dataloader
    
    def fit(self, patience: int = 20, verbose_freq: int = 5) -> None:
        """
        Train the model with improved strategies for small datasets and class imbalance.
        """
        print(f"\n🚀 Starting Training with Class Balance Strategies...")
        print(f"Early stopping patience: {patience} epochs")
        print(f"Verbose output every {verbose_freq} epochs")
        print("-" * 60)
        
        # Training history
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'lr': [],
            'train_f1': [],
            'test_f1': []
        }
        
        # Create balanced dataloader
        dataloader = self._create_balanced_dataloader()
        
        # Early stopping variables
        best_f1 = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        best_model_state = None
        
        # Prepare test data
        if len(self.data_split['X_test']) > 0:
            X_test = torch.tensor(self.data_split['X_test'], dtype=torch.float32).to(self.device)
            y_test = torch.tensor(self.label_encoder.transform(self.data_split['y_test']), dtype=torch.long).to(self.device)
        else:
            X_test = None
            y_test = None
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            total_loss = 0
            all_preds = []
            all_targets = []
            
            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                
                # Clear memory
                del X_batch, y_batch, outputs, loss
            
            # Calculate training metrics
            train_accuracy = accuracy_score(all_targets, all_preds)
            train_f1 = f1_score(all_targets, all_preds, average='macro')
            avg_loss = total_loss / len(dataloader)
            epoch_time = time.time() - epoch_start_time
            
            # Evaluation phase
            test_accuracy = 0.0
            test_f1 = 0.0
            if X_test is not None:
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model(X_test)
                    _, test_predicted = torch.max(test_outputs, 1)
                    test_accuracy = accuracy_score(y_test.cpu().numpy(), test_predicted.cpu().numpy())
                    test_f1 = f1_score(y_test.cpu().numpy(), test_predicted.cpu().numpy(), average='macro')
            
            # Update learning rate scheduler
            self.scheduler.step(test_f1 if X_test is not None else train_f1)
            
            # Store training history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_loss)
            self.training_history['train_acc'].append(train_accuracy)
            self.training_history['test_acc'].append(test_accuracy)
            self.training_history['train_f1'].append(train_f1)
            self.training_history['test_f1'].append(test_f1)
            self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            if (epoch + 1) % verbose_freq == 0 or epoch == 0 or epoch == self.epochs - 1:
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / (epoch + 1)) * (self.epochs - epoch - 1)
                
                print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.4f} | "
                      f"Test Acc: {test_accuracy:.4f} | "
                      f"Train F1: {train_f1:.4f} | "
                      f"Test F1: {test_f1:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Early stopping based on F1 score (better for imbalanced data)
            current_f1 = test_f1 if X_test is not None else train_f1
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_model_state = self.model.state_dict().copy()
                if (epoch + 1) % verbose_freq == 0 or epoch == 0:
                    print(f"  ✅ New best F1: {best_f1:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        total_time = time.time() - start_time
        print(f"\n🎯 Training completed in {total_time/60:.1f} minutes")
        print(f"Best model from epoch {best_epoch} with F1 score {best_f1:.4f}")
        
        # Plot training history
        self._plot_training_history()
    
    def _plot_training_history(self):
        """Plot comprehensive training history."""
        if not hasattr(self, 'training_history'):
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_history['epoch']
        
        # Loss curve
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        if any(acc > 0 for acc in self.training_history['test_acc']):
            ax2.plot(epochs, self.training_history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score curves (important for imbalanced data)
        ax3.plot(epochs, self.training_history['train_f1'], 'g-', label='Training F1', linewidth=2)
        if any(f1 > 0 for f1 in self.training_history['test_f1']):
            ax3.plot(epochs, self.training_history['test_f1'], 'orange', label='Test F1', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score Curves (Class Balance Metric)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate
        ax4.plot(epochs, self.training_history['lr'], 'm-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Predict with preprocessing."""
        X_processed = self.preprocessor.transform(X)
        
        self.model.eval()
        all_preds = []
        
        for i in range(0, len(X_processed), batch_size):
            batch_X = X_processed[i:i+batch_size]
            X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
            
            del X_tensor, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.concatenate(all_preds)
    
    def predict_labels(self, X: np.ndarray, batch_size: int = 64) -> List[str]:
        """Predict and decode to original labels."""
        preds_encoded = self.predict(X, batch_size)
        return self.label_encoder.inverse_transform(preds_encoded)
    
    def evaluate(self, batch_size: int = 64) -> Dict[str, Any]:
        """Comprehensive evaluation with class-wise metrics."""
        if len(self.data_split['X_test']) == 0:
            print("No test data available for evaluation.")
            return {}
            
        predictions_labels = self.predict_labels(self.data_split['X_test'], batch_size)
        y_true = self.data_split['y_test']
        
        accuracy = accuracy_score(y_true, predictions_labels)
        precision_macro = precision_score(y_true, predictions_labels, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, predictions_labels, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, predictions_labels, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, predictions_labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, predictions_labels, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, predictions_labels, average='weighted', zero_division=0)
        
        print("=== Improved Spectral Transformer Evaluation ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"Macro Precision: {precision_macro:.4f}")
        print(f"Macro Recall: {recall_macro:.4f}")
        
        return {
            'classification': {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted
            }
        }
    
    def plot_confusion_matrix(self, show_plot: bool = True, batch_size: int = 64) -> np.ndarray:
        """Plot detailed confusion matrix with class balance information."""
        if len(self.data_split['X_test']) == 0:
            print("No test data available for confusion matrix.")
            return np.array([])
            
        y_pred_labels = self.predict_labels(self.data_split['X_test'], batch_size)
        y_true = self.data_split['y_test']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_labels, labels=self.label_encoder.classes_)
        
        # Print detailed confusion matrix
        print("\n=== Improved Spectral Transformer Confusion Matrix ===")
        print("Predicted ->")
        print("Actual |", " | ".join(f"{label:>8}" for label in self.label_encoder.classes_))
        print("-" * (10 + 10 * len(self.label_encoder.classes_)))
        for i, true_label in enumerate(self.label_encoder.classes_):
            row = [f"{cm[i, j]:>8}" for j in range(len(self.label_encoder.classes_))]
            print(f"{true_label:>6} | {' | '.join(row)}")
        
        # Calculate per-class metrics
        print("\n=== Per-Class Performance ===")
        for i, class_name in enumerate(self.label_encoder.classes_):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        if show_plot:
            # Generate classification report
            report = classification_report(y_true, y_pred_labels, target_names=self.label_encoder.classes_, zero_division=0)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Calculate total for percentages
            total = cm.sum()
            
            # Create annotation array with counts and percentages
            annot = np.array([[f"{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)" 
                             for j in range(cm.shape[1])] 
                            for i in range(cm.shape[0])])
            
            # Plot heatmap on ax1
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                       xticklabels=self.label_encoder.classes_, 
                       yticklabels=self.label_encoder.classes_, ax=ax1)
            ax1.set_title('Improved Spectral Transformer\nConfusion Matrix')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Plot classification report on ax2
            ax2.text(0.1, 0.5, report, fontsize=10, verticalalignment='center', fontfamily='monospace')
            ax2.set_title('Classification Report')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return cm
    
    def predict_new_data(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None, 
                        show_confusion_matrix: bool = False, batch_size: int = 64) -> Dict[str, Any]:
        """Predict on new data with comprehensive evaluation."""
        predictions_labels = self.predict_labels(X_new, batch_size)
        result = {'predictions_labels': predictions_labels, 'n_predictions': len(predictions_labels)}
        
        print(f"=== Improved Spectral Transformer Predictions ===")
        print(f"Number of new samples: {len(predictions_labels)}")
        
        # Show prediction distribution
        pred_dist = dict(zip(*np.unique(predictions_labels, return_counts=True)))
        print(f"Predicted labels distribution: {pred_dist}")
        
        if y_new is not None:
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_new, predictions_labels)
            precision_macro = precision_score(y_new, predictions_labels, average='macro', zero_division=0)
            recall_macro = recall_score(y_new, predictions_labels, average='macro', zero_division=0)
            f1_macro = f1_score(y_new, predictions_labels, average='macro', zero_division=0)
            
            precision_weighted = precision_score(y_new, predictions_labels, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_new, predictions_labels, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_new, predictions_labels, average='weighted', zero_division=0)
            
            # Per-class metrics
            precision_per_class = precision_score(y_new, predictions_labels, average=None, zero_division=0)
            recall_per_class = recall_score(y_new, predictions_labels, average=None, zero_division=0)
            f1_per_class = f1_score(y_new, predictions_labels, average=None, zero_division=0)
            
            result['evaluation'] = {
                'classification': {
                    'accuracy': accuracy,
                    'precision_macro': precision_macro,
                    'recall_macro': recall_macro,
                    'f1_macro': f1_macro,
                    'precision_weighted': precision_weighted,
                    'recall_weighted': recall_weighted,
                    'f1_weighted': f1_weighted,
                    'precision_per_class': dict(zip(self.label_encoder.classes_, precision_per_class)),
                    'recall_per_class': dict(zip(self.label_encoder.classes_, recall_per_class)),
                    'f1_per_class': dict(zip(self.label_encoder.classes_, f1_per_class))
                }
            }
            
            print(f"\n=== Evaluation Metrics ===")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {f1_macro:.4f}")
            print(f"Weighted F1: {f1_weighted:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(y_new, predictions_labels, target_names=self.label_encoder.classes_, zero_division=0))
            
            if show_confusion_matrix:
                cm = confusion_matrix(y_new, predictions_labels, labels=self.label_encoder.classes_)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                total = cm.sum()
                annot = np.array([[f"{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)" 
                                 for j in range(cm.shape[1])] 
                                for i in range(cm.shape[0])])
                
                sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                           xticklabels=self.label_encoder.classes_, 
                           yticklabels=self.label_encoder.classes_, ax=ax1)
                ax1.set_title('Improved Spectral Transformer\nConfusion Matrix (New Data)')
                ax1.set_ylabel('True Label')
                ax1.set_xlabel('Predicted Label')
                
                report = classification_report(y_new, predictions_labels, target_names=self.label_encoder.classes_, zero_division=0)
                ax2.text(0.1, 0.5, report, fontsize=10, verticalalignment='center', fontfamily='monospace')
                ax2.set_title('Classification Report (New Data)')
                ax2.axis('off')
                
                plt.tight_layout()
                plt.show()
        
        return result
    
    def save_model(self, filepath: str) -> None:
        """Save the complete model with preprocessing parameters."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'preprocessor': self.preprocessor,
            'data_split_raw': self.data_split_raw,
            'model_config': {
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
                'use_class_weights': self.use_class_weights,
                'use_weighted_sampling': self.use_weighted_sampling
            },
            'training_history': getattr(self, 'training_history', None)
        }, filepath)
        print(f"Improved Spectral Transformer model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'SpectralTransformerModel':
        """Load the complete model with preprocessing parameters."""
        checkpoint = torch.load(filepath)
        
        # Reconstruct instance
        instance = SpectralTransformerModel(
            checkpoint['data_split_raw'],
            **checkpoint['model_config']
        )
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.label_encoder = checkpoint['label_encoder']
        instance.preprocessor = checkpoint['preprocessor']
        instance.training_history = checkpoint.get('training_history', None)
        return instance
    

# Initialize and train the Spectral Transformer model with memory optimizations
spectral_transformer = SpectralTransformerModel(
    data_split=data_split,
    d_model=32,       # INCREASE from 8 to 32 (middle ground)
    n_heads=4,        # INCREASE back to 4  
    n_layers=1,       # Keep at 1 (good)
    lr=0.0005,        # INCREASE learning rate slightly
    dropout=0.4,      # DECREASE from 0.6 to 0.4
    batch_size=32,    # DECREASE for better learning
    weight_decay=0.05, # DECREASE regularization
    epoch=100
) 

# Train the model
spectral_transformer.fit(patience=50, verbose_freq=5)

# Evaluate on test set
spectral_transformer.evaluate()

# Plot confusion matrix
cm = spectral_transformer.plot_confusion_matrix()

# Predict on new data
SPECTRAL_TRANSFORMER_PREDICT_EVALUATION = spectral_transformer.predict_new_data(
    data_split_predict['X_train'], 
    data_split_predict['y_train'], 
    show_confusion_matrix=True
)["evaluation"]

X_train = data_split['X_train']
y_train = data_split['y_train']
print("Class distribution:", np.unique(y_train, return_counts=True))
print("Data statistics:", X_train.mean(axis=0)[:10], X_train.std(axis=0)[:10])
print("Value ranges:", X_train.min(), X_train.max())
print("Problematic values:", np.isnan(X_train).sum(), np.isinf(X_train).sum())