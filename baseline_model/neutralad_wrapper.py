"""
NeuTraL-AD Wrapper - Tích hợp NeuTraL-AD vào baseline với interface giống PyOD
Neural Transformation Learning for Anomaly Detection
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TabTransformNet(nn.Module):
    """Transformation network cho tabular data"""
    def __init__(self, x_dim, h_dim, num_layers):
        super(TabTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            net.append(nn.Linear(input_dim, h_dim, bias=False))
            net.append(nn.ReLU())
            input_dim = h_dim
        net.append(nn.Linear(input_dim, x_dim, bias=False))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class TabEncoder(nn.Module):
    """Encoder network cho tabular data"""
    def __init__(self, x_dim, h_dim, z_dim, bias=False, num_layers=5, batch_norm=False):
        super(TabEncoder, self).__init__()
        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, h_dim, bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(h_dim, affine=bias))
            enc.append(nn.ReLU())
            input_dim = h_dim
        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_dim, z_dim, bias=bias)

    def forward(self, x):
        z = self.enc(x)
        z = self.fc(z)
        return z


class NeuTraLADModel(nn.Module):
    """NeuTraL-AD model cho tabular data"""
    def __init__(self, x_dim, z_dim=32, h_dim=32, num_trans=11, 
                 trans_hdim=None, trans_nlayers=2, enc_nlayers=5,
                 enc_bias=False, batch_norm=False, trans_type='residual', device='cuda'):
        super(NeuTraLADModel, self).__init__()
        
        if trans_hdim is None:
            trans_hdim = x_dim
            
        self.num_trans = num_trans
        self.trans_type = trans_type
        self.device = device
        self.z_dim = z_dim
        
        # Encoder
        self.enc = TabEncoder(x_dim, h_dim, z_dim, enc_bias, enc_nlayers, batch_norm)
        
        # Transformations
        self.trans = nn.ModuleList([
            TabTransformNet(x_dim, trans_hdim, trans_nlayers) 
            for _ in range(num_trans)
        ])

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0], self.num_trans, x.shape[-1]).to(x)
        
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
                
        x_cat = torch.cat([x.unsqueeze(1), x_T], 1)
        zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.num_trans + 1, self.z_dim)
        return zs


class DCLLoss(nn.Module):
    """Debiased Contrastive Loss"""
    def __init__(self, temperature=0.1):
        super(DCLLoss, self).__init__()
        self.temp = temperature

    def forward(self, z, eval=False):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n, z
        z_trans = z[:, 1:]  # n, k-1, z
        batch_size, num_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1)) / self.temp)  # n, k, k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n, k-1

        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp)  # n, k-1
        K = num_trans - 1
        scale = 1 / np.abs(K * np.log(1.0 / K))

        loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale

        if eval:
            return loss_tensor.sum(1)
        else:
            return loss_tensor.sum(1)



class NeuTraLAD:
    """
    NeuTraL-AD model wrapper với interface tương thích PyOD
    
    Neural Transformation Learning for Deep Anomaly Detection Beyond Images
    Reference: https://arxiv.org/abs/2103.16440
    """
    def __init__(self, latent_dim=32, enc_hdim=32, enc_nlayers=5, 
                 num_trans=11, trans_nlayers=2, trans_hdim=None,
                 trans_type='residual', enc_bias=False, batch_norm=False,
                 num_epochs=200, batch_size=128, lr=1e-3, l2=1e-5,
                 loss_temp=0.1, device=None, verbose=0):
        """
        Parameters:
            latent_dim: Kích thước latent space (z_dim)
            enc_hdim: Hidden dimension của encoder
            enc_nlayers: Số layers của encoder
            num_trans: Số lượng transformations
            trans_nlayers: Số layers của transformation network
            trans_hdim: Hidden dimension của transformation (None = x_dim)
            trans_type: Loại transformation ('residual', 'mul', 'forward')
            enc_bias: Sử dụng bias trong encoder
            batch_norm: Sử dụng batch normalization
            num_epochs: Số epochs training
            batch_size: Batch size
            lr: Learning rate
            l2: L2 regularization weight
            loss_temp: Temperature cho contrastive loss
            device: 'cuda' hoặc 'cpu'
            verbose: 0 = silent, 1 = progress
        """
        self.latent_dim = latent_dim
        self.enc_hdim = enc_hdim
        self.enc_nlayers = enc_nlayers
        self.num_trans = num_trans
        self.trans_nlayers = trans_nlayers
        self.trans_hdim = trans_hdim
        self.trans_type = trans_type
        self.enc_bias = enc_bias
        self.batch_norm = batch_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.loss_temp = loss_temp
        self.verbose = verbose
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.loss_fun = None
        self.in_shape = None

    def fit(self, X, y=None):
        """
        Fit NeuTraL-AD model
        
        Parameters:
            X: Training data (numpy array), shape (n_samples, n_features)
            y: Labels (không sử dụng, chỉ để tương thích interface)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        self.in_shape = X.shape[1]
        
        # Determine trans_hdim
        trans_hdim = self.trans_hdim if self.trans_hdim else self.in_shape
        
        # Initialize model
        self.model = NeuTraLADModel(
            x_dim=self.in_shape,
            z_dim=self.latent_dim,
            h_dim=self.enc_hdim,
            num_trans=self.num_trans,
            trans_hdim=trans_hdim,
            trans_nlayers=self.trans_nlayers,
            enc_nlayers=self.enc_nlayers,
            enc_bias=self.enc_bias,
            batch_norm=self.batch_norm,
            trans_type=self.trans_type,
            device=self.device
        ).to(self.device)
        
        # Loss function
        self.loss_fun = DCLLoss(temperature=self.loss_temp)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)
        
        # DataLoader
        X_tensor = torch.FloatTensor(X)
        train_dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for batch in train_loader:
                samples = batch[0].to(self.device)
                
                optimizer.zero_grad()
                z = self.model(samples)
                loss = self.loss_fun(z)
                loss_mean = loss.mean()
                loss_mean.backward()
                optimizer.step()
                
                epoch_loss += loss.sum().item()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader.dataset)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
        
        return self

    def decision_function(self, X):
        """
        Compute anomaly scores
        
        Parameters:
            X: Test data (numpy array), shape (n_samples, n_features)
            
        Returns:
            scores: Anomaly scores (higher = more anomalous)
        """
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X)
        test_dataset = TensorDataset(X_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        scores = []
        with torch.no_grad():
            for batch in test_loader:
                samples = batch[0].to(self.device)
                z = self.model(samples)
                score = self.loss_fun(z, eval=True)
                scores.append(score.cpu().numpy())
        
        return np.concatenate(scores)

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
        scores_min = scores.min()
        scores_max = scores.max()
        scores_normalized = (scores - scores_min) / (scores_max - scores_min + 1e-10)
        
        # Return probabilities for [normal, anomaly]
        proba = np.zeros((len(scores), 2))
        proba[:, 1] = scores_normalized  # Probability of anomaly
        proba[:, 0] = 1 - scores_normalized  # Probability of normal
        
        return proba
    
    def fit_predict(self, X, y=None):
        """
        Fit model và predict labels
        
        Parameters:
            X: Training/test data
            y: Labels (không sử dụng)
            
        Returns:
            labels: Binary labels
        """
        self.fit(X, y)
        return self.predict(X)
