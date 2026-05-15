import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    """
    One-class training: labels không dùng cũng được.
    Giữ format gần giống template của bạn.
    """
    def __init__(self, data, labels=None):
        self.data = np.asarray(data, dtype=np.float32)
        if labels is None:
            labels = np.zeros(len(self.data), dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = torch.from_numpy(self.data[idx]).float()
        y = torch.tensor(self.labels[idx]).float()
        return x, y, torch.tensor([0])  # dummy giống template


def random_orthogonal_vectors(num_vectors: int, vector_dim: int) -> np.ndarray:
    """Gram–Schmidt để tạo các vector trực giao (giống code gốc)."""
    while True:
        random_matrix = np.random.randn(num_vectors, vector_dim)
        if np.linalg.matrix_rank(random_matrix) == num_vectors:
            break

    orthogonal_vectors = np.zeros((num_vectors, vector_dim), dtype=np.float32)
    for i in range(num_vectors):
        v = random_matrix[i].astype(np.float32)
        for j in range(i):
            v = v - np.dot(v, orthogonal_vectors[j]) * orthogonal_vectors[j]
        orthogonal_vectors[i] = v / (np.linalg.norm(v) + 1e-12)
    return orthogonal_vectors


class DRLAD(nn.Module):
    """
    DRL-AD (Decomposition Representation Learning for Anomaly Detection)
    - forward()/score: trả anomaly score (mse giữa h và h_)
    - fit: train one-class trên X (normal samples)
    - decision_function: trả score numpy (higher = more anomalous)
    """
    name = "DRL-AD"

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        en_nlayers: int = 3,
        de_nlayers: int = 3,   # giữ param cho giống config, dù decoder bản gốc chỉ 1 linear
        basis_vector_num: int = 5,
        diversity: bool = True,
        plearn: bool = False,

        # losses
        input_info: bool = True,
        input_info_ratio: float = 0.1,
        cl: bool = True,
        cl_ratio: float = 0.06,

        # training
        epochs: int = 200,
        learning_rate: float = 0.05,
        sche_gamma: float = 0.98,
        batch_size: int = 512,
        weight_decay: float = 1e-5,

        # misc
        device: str = None,
        random_seed: int = 42,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.en_nlayers = int(en_nlayers)
        self.de_nlayers = int(de_nlayers)
        self.basis_vector_num = int(basis_vector_num)
        self.diversity = bool(diversity)
        self.plearn = bool(plearn)

        self.input_info = bool(input_info)
        self.input_info_ratio = float(input_info_ratio)
        self.cl = bool(cl)
        self.cl_ratio = float(cl_ratio)

        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.sche_gamma = float(sche_gamma)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.random_seed = int(random_seed)

        self._build_network()
        self.to(self.device)

    def _set_seed(self, seed: int):
        seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_network(self):
        # ----- basis vectors -----
        if not self.diversity:
            init_b = np.random.rand(self.basis_vector_num, self.hidden_dim).astype(np.float32)
        else:
            init_b = random_orthogonal_vectors(self.basis_vector_num, self.hidden_dim)

        self.basis_vector = nn.Parameter(
            torch.tensor(init_b, dtype=torch.float32),
            requires_grad=self.plearn
        )

        # ----- phi(x) -> weights -----
        phi_layers = []
        encoder_dim = self.in_features
        for _ in range(self.en_nlayers - 2):
            phi_layers.append(nn.Linear(encoder_dim, self.hidden_dim, bias=False))
            phi_layers.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        phi_layers.append(nn.Linear(encoder_dim, self.basis_vector_num, bias=False))
        self.phi = nn.Sequential(*phi_layers)

        # ----- encoder -----
        enc_layers = []
        encoder_dim = self.in_features
        for _ in range(self.en_nlayers - 1):
            enc_layers.append(nn.Linear(encoder_dim, self.hidden_dim, bias=False))
            enc_layers.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        self.encoder = nn.Sequential(*enc_layers)

        # ----- decoder (bản gốc chỉ 1 linear) -----
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.in_features, bias=False)
        )

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Score chính: mse giữa h=encoder(x) và h_=softmax(phi(x)) @ basis_vector
        Trả tensor shape (B,)
        """
        X = X.to(self.device)

        h = self.encoder(X)  # (B, hidden)
        weight = F.softmax(self.phi(X), dim=1)  # (B, K)
        h_ = weight @ self.basis_vector  # (B, hidden)

        mse = F.mse_loss(h, h_, reduction='none')  # (B, hidden)
        s = mse.sum(dim=1)  # (B,)
        return s

    def forward(self, X: torch.Tensor):
        return self.score(X)

    def fit(self, X, y=None, seed: int = None):
        """
        Train one-class trên X (normal-only).
        """
        if seed is None:
            seed = self.random_seed
        self._set_seed(seed)

        dataset = CustomDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)

        self.train()
        for _ in range(self.epochs):
            for x_input, _, _ in train_loader:
                x_input = x_input.to(self.device)

                # decomposition loss
                loss = self.score(x_input).mean()

                # alignment loss
                if self.input_info:
                    h = self.encoder(x_input)
                    x_tilde = self.decoder(h)
                    s_loss = F.cosine_similarity(x_tilde, x_input, dim=-1).mean() * (-1.0)
                    loss = loss + self.input_info_ratio * s_loss

                # separation loss (contrastive-like)
                if self.cl:
                    w = F.softmax(self.phi(x_input), dim=1)  # (B,K)
                    B = w.shape[0]
                    m = max(2, int(B * 0.8))
                    idx = torch.randperm(B, device=self.device)[:m]
                    w = w[idx]  # (m,K)

                    num = w @ w.T
                    den = (w.norm(dim=1, keepdim=True) @ w.norm(dim=1, keepdim=True).T).clamp_min(1e-12)
                    cos = num / den

                    eye = torch.eye(m, device=self.device)
                    d_loss = ((1.0 - eye) * cos).sum() / (m * m)
                    loss = loss + self.cl_ratio * d_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        return self

    @torch.no_grad()
    def decision_function(self, X):
        """
        Trả anomaly scores numpy (higher = more anomalous).
        """
        self.eval()
        X = np.asarray(X, dtype=np.float32)
        dl = DataLoader(CustomDataset(X), batch_size=self.batch_size, shuffle=False, num_workers=0)

        scores = []
        for x_input, _, _ in dl:
            x_input = x_input.to(self.device)
            s = self.score(x_input).detach().cpu().numpy()
            scores.append(s)

        return np.concatenate(scores, axis=0).astype(np.float32)

    def predict(self, X, percentile: float = 95.0):
        """
        Predict nhị phân theo percentile (mặc định 95).
        Benchmark của bạn đang dùng AUROC/AUCPR nên thường không cần predict().
        """
        scores = self.decision_function(X)
        thr = np.percentile(scores, float(percentile))
        return (scores > thr).astype(int)
