"""
DASVDD Wrapper - Tích hợp DASVDD vào baseline với interface giống PyOD
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class SimpleAutoEncoder(nn.Module):
    """
    Simple AutoEncoder cho tabular data (IoT datasets)
    """
    def __init__(self, input_shape, code_size=32):
        super().__init__()
        
        # Tính toán hidden sizes
        hidden1 = max(64, input_shape // 2)
        hidden2 = max(32, input_shape // 4)
        
        # Encoder
        self.encoder_hidden_layer = nn.Linear(input_shape, hidden1)
        self.encoder_middle = nn.Linear(hidden1, hidden2)
        self.encoder_output_layer = nn.Linear(hidden2, code_size)
        
        # Decoder
        self.decoder_hidden_layer = nn.Linear(code_size, hidden2)
        self.decoder_middle = nn.Linear(hidden2, hidden1)
        self.decoder_output_layer = nn.Linear(hidden1, input_shape)
    
    def forward(self, features):
        # Encoder
        activation = F.leaky_relu(self.encoder_hidden_layer(features))
        activation = F.leaky_relu(self.encoder_middle(activation))
        code = F.leaky_relu(self.encoder_output_layer(activation))
        
        # Decoder
        activation = F.leaky_relu(self.decoder_hidden_layer(code))
        activation = F.leaky_relu(self.decoder_middle(activation))
        reconstructed = torch.sigmoid(self.decoder_output_layer(activation))
        
        return reconstructed, code


class DASVDD:
    """
    DASVDD model wrapper với interface tương thích PyOD
    """
    def __init__(self, code_size=32, num_epochs=100, batch_size=128, 
                 lr=1e-3, K=0.9, T=10, device=None, verbose=0):
        """
        Parameters:
            code_size: Kích thước latent space
            num_epochs: Số epochs training
            batch_size: Batch size
            lr: Learning rate
            K: Tỷ lệ samples dùng cho SVDD loss
            T: Số lần tune gamma
            device: 'cuda' hoặc 'cpu'
            verbose: 0 = silent, 1 = progress
        """
        self.code_size = code_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.K = K
        self.T = T
        self.verbose = verbose
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.C = None  # Center
        self.Gamma = None
        self.criterion = nn.MSELoss()
        self.in_shape = None
    
    def _tune_gamma(self, train_loader):
        """Tune gamma parameter"""
        gamma = 0
        for k in range(self.T):
            temp_model = SimpleAutoEncoder(self.in_shape, self.code_size).to(self.device)
            R = 0
            RE = 0
            
            for batch_features in train_loader:
                if isinstance(batch_features, list):
                    batch_features = batch_features[0]
                batch_features = batch_features.view(-1, self.in_shape).to(self.device)
                outputs, code = temp_model(batch_features)
                R += torch.sum((code.to(self.device)) ** 2, dim=1)[0]
                RE += self.criterion(outputs, batch_features)
            
            R = R / len(train_loader)
            RE = RE / len(train_loader)
            gamma += RE / R
        
        gamma = gamma / self.T
        return gamma.detach().item()
    
    def fit(self, X, y=None):
        """
        Fit DASVDD model
        
        Parameters:
            X: Training data (numpy array)
            y: Labels (không sử dụng, chỉ để tương thích interface)
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        train_dataset = torch.utils.data.TensorDataset(X_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        self.in_shape = X.shape[1]
        
        # Initialize model
        self.model = SimpleAutoEncoder(self.in_shape, self.code_size).to(self.device)
        
        # Initialize center C
        self.C = torch.randn(self.code_size, device=self.device, requires_grad=True)
        
        # Optimizers
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        update_center = torch.optim.Adagrad([self.C], lr=1, lr_decay=0.01)
        
        # Tune gamma
        if self.verbose:
            print("Tuning gamma...")
        self.Gamma = self._tune_gamma(train_loader)
        if self.verbose:
            print(f"Gamma: {self.Gamma:.4f}")
        
        # Training
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            
            for batch_features in train_loader:
                if isinstance(batch_features, list):
                    batch_features = batch_features[0]
                batch_features = batch_features.view(-1, self.in_shape).to(self.device)
                
                Num_batch = int(np.ceil(self.K * batch_features.size()[0]))
                
                # Train autoencoder + SVDD
                optimizer.zero_grad()
                outputs, code = self.model(batch_features[:Num_batch, :])
                R = torch.sum((code.to(self.device) - self.C) ** 2, dim=1)[0]
                train_loss = self.criterion(outputs, batch_features[:Num_batch, :]) + self.Gamma * R
                train_loss.backward()
                optimizer.step()
                
                epoch_loss += train_loss.item()
                
                # Update center
                _, c_code = self.model(batch_features[Num_batch:, :])
                center = torch.mean(c_code, axis=0)
                center_loss = self.criterion(self.C, center)
                center_loss.backward()
                update_center.step()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss / len(train_loader):.6f}")
        
        return self
    
    def decision_function(self, X):
        """
        Compute anomaly scores
        
        Parameters:
            X: Test data (numpy array)
            
        Returns:
            scores: Anomaly scores (higher = more anomalous)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        test_dataset = torch.utils.data.TensorDataset(X_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False
        )
        
        scores = []
        with torch.no_grad():
            for x_test in test_loader:
                if isinstance(x_test, list):
                    x_test = x_test[0]
                x_test = x_test.view(-1, self.in_shape).to(self.device)
                x_test_hat, code_test = self.model(x_test)
                
                # Compute anomaly score
                loss = self.criterion(x_test_hat, x_test) + \
                       self.Gamma * torch.sum((code_test.to(self.device) - self.C) ** 2, dim=1)[0]
                scores.append(loss.to("cpu").item())
        
        return np.array(scores)
    
    def predict(self, X):
        """
        Predict labels (0 = normal, 1 = anomaly)
        
        Parameters:
            X: Test data (numpy array)
            
        Returns:
            labels: Binary labels
        """
        scores = self.decision_function(X)
        # Sử dụng median làm threshold
        threshold = np.median(scores)
        return (scores > threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Predict probability estimates
        
        Parameters:
            X: Test data (numpy array)
            
        Returns:
            proba: Probability estimates [P(normal), P(anomaly)]
        """
        scores = self.decision_function(X)
        # Normalize scores to [0, 1]
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        # Return probabilities for [normal, anomaly]
        proba = np.zeros((len(scores), 2))
        proba[:, 1] = scores_normalized  # Probability of anomaly
        proba[:, 0] = 1 - scores_normalized  # Probability of normal
        
        return proba
