"""
Deep iForest (DIF) Wrapper - Fixed version với robust random_state handling

Wrapper này fix lỗi random_state bằng cách:
1. Validate input parameters
2. Monkey patch DIFBase để ensure random_state luôn hợp lệ
3. Convert numpy types sang Python native types
"""
import numpy as np
import sys
import os

# Setup import path
try:
    from algorithms.dif import DIF as DIFBase
except ModuleNotFoundError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        current_dir,
        os.path.join(current_dir, '.'), 

        os.path.join(current_dir, '..', 'deep-iforest'),
        os.path.join(current_dir, '..', '..', 'deep-iforest'),
        r'D:\UIT\Research\Duongcpmputer\baselinePlus\deep-iforest',
    ]
    
    deep_iforest_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, 'algorithms')):
            deep_iforest_path = abs_path
            break
    
    if deep_iforest_path is None:
        raise ImportError("Cannot find deep-iforest folder!")
    
    sys.path.insert(0, deep_iforest_path)
    from algorithms.dif import DIF as DIFBase


def validate_random_state(random_state):
    """
    Validate và convert random_state thành kiểu hợp lệ
    
    Returns:
        int, float, str, bytes, bytearray, or None
    """
    if random_state is None:
        return None
    
    # Convert numpy integer types
    try:
        if isinstance(random_state, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(random_state)
    except AttributeError:
        pass
    
    # Convert numpy floating types (compatible with NumPy 2.0)
    try:
        if isinstance(random_state, (np.floating, np.float16, np.float32, np.float64)):
            return int(random_state)
    except AttributeError:
        pass
    
    # Check if already valid type
    if isinstance(random_state, (int, float, str, bytes, bytearray)):
        return random_state
    
    # Try to convert to int
    try:
        return int(random_state)
    except (TypeError, ValueError):
        pass
    
    # Fallback to default
    print(f"Warning: Invalid random_state type {type(random_state)}, using default 42")
    return 42


class DIFPatched(DIFBase):
    """
    Patched version của DIF để fix random_state issues
    """
    def __init__(self, *args, **kwargs):
        # Validate random_state trước khi pass vào parent
        if 'random_state' in kwargs:
            kwargs['random_state'] = validate_random_state(kwargs['random_state'])
        super().__init__(*args, **kwargs)
    
    def fit(self, X, y=None):
        """Override fit để ensure random_state hợp lệ trong IsolationForest"""
        from sklearn.ensemble import IsolationForest
        import time
        from tqdm import tqdm
        
        # Validate lại random_state trước khi fit
        if hasattr(self, 'random_state'):
            object.__setattr__(self, 'random_state', validate_random_state(self.random_state))
        
        # Call parent's fit logic nhưng patch IsolationForest creation
        start_time = time.time()
        self.n_features = X.shape[-1] if self.data_type != 'graph' else max(X.num_features, 1)
        
        # Generate ensemble seeds và convert sang int
        ensemble_seeds = np.random.randint(0, 1e+5, self.n_ensemble)
        ensemble_seeds = [int(seed) for seed in ensemble_seeds]  # Convert to Python int!
        
        if self.verbose >= 2:
            net = self.Net(n_features=self.n_features, **self.network_args)
            print(net)
        
        self._training_transfer(X, ensemble_seeds)
        
        if self.verbose >= 2:
            it = tqdm(range(self.n_ensemble), desc='clf fitting', ncols=80)
        else:
            it = range(self.n_ensemble)
        
        # Create IsolationForest với validated random_state
        for i in it:
            self.clf_lst.append(
                IsolationForest(
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    random_state=ensemble_seeds[i]  # Đã là Python int
                )
            )
            self.clf_lst[i].fit(self.x_reduced_lst[i])
        
        if self.verbose >= 1:
            print(f'training done, time: {time.time()-start_time:.1f}')
        
        return self


class DIF:
    """
    Deep Isolation Forest (DIF) wrapper với interface tương thích PyOD
    
    Fixed version với robust random_state handling
    
    Parameters:
        n_ensemble: int (default=50)
            Số lượng representations trong ensemble
            
        n_estimators: int (default=6)
            Số lượng isolation trees cho mỗi representation
            
        max_samples: int or float or "auto" (default=256)
            Số samples để train mỗi base estimator
            
        hidden_dim: list (default=[500, 100])
            List các units của hidden layers
            
        rep_dim: int (default=20)
            Dimensionality của representations
            
        activation: str (default='tanh')
            Activation function: {'relu', 'tanh', 'sigmoid', 'leaky_relu'}
            
        batch_size: int (default=64)
            Batch size cho training/inference
            
        device: str (default='cuda')
            Device: 'cuda' hoặc 'cpu'
            
        random_state: int, float, str, None (default=42)
            Random seed
            
        verbose: int (default=0)
            Verbosity level: 0=silent, 1=progress, 2=detailed
    """
    
    def __init__(self, 
                 n_ensemble=50, 
                 n_estimators=6, 
                 max_samples=256,
                 hidden_dim=[500, 100], 
                 rep_dim=20, 
                 skip_connection=None, 
                 dropout=None, 
                 activation='tanh',
                 batch_size=64,
                 new_score_func=True, 
                 new_ensemble_method=True,
                 random_state=42, 
                 device='cuda', 
                 n_processes=1,
                 verbose=0):
        
        # Validate ALL parameters
        random_state = validate_random_state(random_state)
        
        # Convert numpy types in other parameters (NumPy 2.0 compatible)
        def to_int(val):
            """Convert to int, handling numpy types"""
            if isinstance(val, (int, type(None))):
                return val
            try:
                if isinstance(val, np.integer):
                    return int(val)
            except (AttributeError, TypeError):
                pass
            try:
                return int(val)
            except (TypeError, ValueError):
                return val
        
        n_ensemble = to_int(n_ensemble)
        n_estimators = to_int(n_estimators)
        max_samples = to_int(max_samples) if not isinstance(max_samples, str) else max_samples
        batch_size = to_int(batch_size)
        rep_dim = to_int(rep_dim)
        n_processes = to_int(n_processes)
        verbose = to_int(verbose)
        
        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.hidden_dim = hidden_dim
        self.rep_dim = rep_dim
        self.skip_connection = skip_connection
        self.dropout = dropout
        self.activation = activation
        self.batch_size = batch_size
        self.new_score_func = new_score_func
        self.new_ensemble_method = new_ensemble_method
        self.random_state = random_state
        self.device = device
        self.n_processes = n_processes
        self.verbose = verbose
        
        # Khởi tạo patched model
        self.model = DIFPatched(
            network_name='mlp',
            n_ensemble=n_ensemble,
            n_estimators=n_estimators,
            max_samples=max_samples,
            hidden_dim=hidden_dim,
            rep_dim=rep_dim,
            skip_connection=skip_connection,
            dropout=dropout,
            activation=activation,
            data_type='tabular',
            batch_size=batch_size,
            new_score_func=new_score_func,
            new_ensemble_method=new_ensemble_method,
            random_state=random_state,
            device=device,
            n_processes=n_processes,
            verbose=verbose
        )
    
    def fit(self, X, y=None):
        """Fit Deep iForest model"""
        X = np.asarray(X, dtype=np.float32)
        self.model.fit(X, y)
        return self
    
    def decision_function(self, X):
        """Compute anomaly scores (higher = more anomalous)"""
        X = np.asarray(X, dtype=np.float32)
        scores = self.model.decision_function(X)
        return scores
    
    def predict(self, X, threshold=None):
        """Predict labels (0 = normal, 1 = anomaly)"""
        scores = self.decision_function(X)
        if threshold is None:
            threshold = np.median(scores)
        return (scores > threshold).astype(int)
    
    def predict_proba(self, X):
        """Predict probability estimates [P(normal), P(anomaly)]"""
        scores = self.decision_function(X)
        scores_min = scores.min()
        scores_max = scores.max()
        scores_normalized = (scores - scores_min) / (scores_max - scores_min + 1e-10)
        
        proba = np.zeros((len(scores), 2))
        proba[:, 1] = scores_normalized
        proba[:, 0] = 1 - scores_normalized
        return proba
    
    def fit_predict(self, X, y=None):
        """Fit model và predict labels"""
        self.fit(X, y)
        return self.predict(X)
