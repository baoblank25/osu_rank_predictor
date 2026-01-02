import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import numpy as np
from pathlib import Path


class SimpleNN(nn.Module):
    """basic feedforward net, nothing fancy"""
    def __init__(self, input_size, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


class EarlyStopping:
    """
    stops training if validation loss stops improving
    saves best weights and restores them at the end
    """
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.best_loss = None
        self.stopped = False
        self.best_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.wait += 1
            if self.verbose and self.wait % 5 == 0:
                print(f"      EarlyStopping: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.stopped = True
        else:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.wait = 0
        
        return self.stopped
    
    def restore_best_weights(self, model):
        """put the best weights back"""
        if self.best_state:
            model.load_state_dict(self.best_state)


class RankStratifiedEnsembleModel:
    """
    Two-mode ensemble for predicting osu! ranks
    
    ranks 1-1000:  uses log scaling (top players have huge gaps)
    ranks 1k-10k:  uses linear scaling (more evenly distributed)
    
    why? because the difference between rank 1 and 100 is massive but
    rank 5000 to 5100 is basically nothing. log scale handles this better
    for top players.
    
    NOTE: only works for ranks 1-10k because thats all the API gives us
    """
    def __init__(self, n_features, device):
        self.device = device
        self.input_size = n_features
        
        #Two sets of models: one for log-scaled, one for linear
        self.models_log = {}      # {range_key: (rf, nn, scaler)} - for ranks 1-1k
        self.models_linear = {}   # {range_key: (rf, nn, scaler)} - for ranks 1k-10k
        self.trained = False
    
    def _get_rank_range_key(self, rank):
        """Determine which rank range a player belongs to"""
        #Constraints: Only support ranks 1-10k
        if rank < 1 or rank > 10000:
            raise ValueError(
                f"RANK CONSTRAINT VIOLATION\n"
                f"   This model only supports ranks 1-10,000\n"
                f"   Requested rank: {int(rank):,}\n\n"
                f"   Reason: osu! API only provides top 10k leaderboard.\n"
                f"   Beyond rank 10k, data cannot be collected due to API rate limiting.\n"
                f"   Training data is limited to ranks 1-10,000."
            )
        
        if rank <= 100:
            return "1-100", 100, "log"
        elif rank <= 500:
            return "100-500", 500, "log"
        elif rank <= 1000:
            return "500-1k", 1000, "log"
        elif rank <= 2000:
            return "1k-2k", 2000, "linear"
        elif rank <= 5000:
            return "2k-5k", 5000, "linear"
        else:
            return "5k-10k", 10000, "linear"
    
    def fit(self, X, y, epochs=50, batch_size=8):
        """Train with dual scaling modes"""
        
        print(f"Original rank range: {y.min():.0f} - {y.max():.0f}\n")
        
        #Prepare data for both modes
        y_log = np.log1p(y)
        
        #Get rank ranges
        ranges_log = {}
        ranges_linear = {}
        
        for i, rank in enumerate(y):
            range_key, _, mode = self._get_rank_range_key(rank)
            if mode == "log":
                if range_key not in ranges_log:
                    ranges_log[range_key] = []
                ranges_log[range_key].append(i)
            else:
                if range_key not in ranges_linear:
                    ranges_linear[range_key] = []
                ranges_linear[range_key].append(i)
        
        #TRAIN LOG-SCALED MODELS (Ranks 1-1k) 
        if ranges_log:
            print(f"Training LOG-SCALED models (ranks 1-1k):\n")
            
            for range_key in sorted(ranges_log.keys()):
                indices = ranges_log[range_key]
                
                if len(indices) < 3:
                    print(f"{range_key}: Only {len(indices)} samples (skipping)")
                    continue
                
                print(f"{range_key} ({len(indices)} players) - LOG MODE")
                
                X_range = X[indices]
                y_range_log = y_log[indices]  #Use log scale
                
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_range)
                
                # Train RF
                rf = RandomForestRegressor(
                    n_estimators=50, max_depth=10, min_samples_split=3,
                    random_state=42, n_jobs=-1
                )
                rf.fit(X_scaled, y_range_log)
                
                nn_model = SimpleNN(self.input_size, dropout=0.2).to(self.device)
                optimizer = optim.Adam(nn_model.parameters(), lr=0.01, weight_decay=1e-5)
                loss_fn = torch.nn.MSELoss()
                early_stop = EarlyStopping(patience=15, min_delta=0.001, verbose=False)
                
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                y_t = torch.FloatTensor(y_range_log).unsqueeze(1).to(self.device)
                y_mean = y_t.mean()
                y_std = y_t.std() + 1e-8
                y_normalized = (y_t - y_mean) / y_std
                
                # Split for validation (80/20)
                n_train = int(len(X_t) * 0.8)
                perm = torch.randperm(len(X_t))
                train_idx, val_idx = perm[:n_train], perm[n_train:]
                
                #Show training progress
                print(f"    Training NN with early stopping (patience=15)...")
                
                for epoch in range(epochs):
                    nn_model.train()
                    indices_t = torch.randperm(len(train_idx))
                    epoch_loss = 0
                    
                    for i in range(0, len(train_idx), batch_size):
                        idx = train_idx[indices_t[i:i+batch_size]]
                        X_batch = X_t[idx]
                        y_batch = y_normalized[idx]
                        
                        optimizer.zero_grad()
                        pred = nn_model(X_batch)
                        loss = loss_fn(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    # check val loss
                    nn_model.eval()
                    with torch.no_grad():
                        val_pred = nn_model(X_t[val_idx])
                        val_loss = loss_fn(val_pred, y_normalized[val_idx]).item()
                    
                    if early_stop(val_loss, nn_model):
                        print(f"      stopped early @ epoch {epoch+1}")
                        early_stop.restore_best_weights(nn_model)
                        break
                    
                    if (epoch + 1) % 20 == 0:
                        print(f"      epoch {epoch+1}/{epochs} - train: {epoch_loss:.4f}, val: {val_loss:.4f}")
                
                self.models_log[range_key] = (rf, nn_model, scaler)
                print(f"done\n")
        
        #TRAIN LINEAR MODELS (Ranks 1k-10k)
        if ranges_linear:
            print(f"Training LINEAR models (ranks 1k-10k):\n")
            
            for range_key in sorted(ranges_linear.keys()):
                indices = ranges_linear[range_key]
                
                if len(indices) < 3:
                    print(f"{range_key}: Only {len(indices)} samples (skipping)")
                    continue
                
                print(f"{range_key} ({len(indices)} players) - LINEAR MODE")
                
                X_range = X[indices]
                y_range = y[indices]
                
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_range)
                
                #Train RF
                rf = RandomForestRegressor(
                    n_estimators=50, max_depth=10, min_samples_split=3,
                    random_state=42, n_jobs=-1
                )
                rf.fit(X_scaled, y_range)
                
                nn_model = SimpleNN(self.input_size, dropout=0.2).to(self.device)
                optimizer = optim.Adam(nn_model.parameters(), lr=0.01, weight_decay=1e-5)
                loss_fn = torch.nn.MSELoss()
                early_stop = EarlyStopping(patience=15, min_delta=0.001, verbose=False)
                
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                y_t = torch.FloatTensor(y_range).unsqueeze(1).to(self.device)
                y_mean = y_t.mean()
                y_std = y_t.std() + 1e-8
                y_normalized = (y_t - y_mean) / y_std
                
                # Split for validation (80/20)
                n_train = int(len(X_t) * 0.8)
                perm = torch.randperm(len(X_t))
                train_idx, val_idx = perm[:n_train], perm[n_train:]
                
                print(f"    Training NN with early stopping (patience=15)...")
                
                for epoch in range(epochs):
                    nn_model.train()
                    indices_t = torch.randperm(len(train_idx))
                    epoch_loss = 0
                    
                    for i in range(0, len(train_idx), batch_size):
                        idx = train_idx[indices_t[i:i+batch_size]]
                        X_batch = X_t[idx]
                        y_batch = y_normalized[idx]
                        
                        optimizer.zero_grad()
                        pred = nn_model(X_batch)
                        loss = loss_fn(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    # val loss check
                    nn_model.eval()
                    with torch.no_grad():
                        val_pred = nn_model(X_t[val_idx])
                        val_loss = loss_fn(val_pred, y_normalized[val_idx]).item()
                    
                    if early_stop(val_loss, nn_model):
                        print(f"      stopped early @ epoch {epoch+1}")
                        early_stop.restore_best_weights(nn_model)
                        break
                    
                    if (epoch + 1) % 20 == 0:
                        print(f"      epoch {epoch+1}/{epochs} - train: {epoch_loss:.4f}, val: {val_loss:.4f}")
                
                self.models_linear[range_key] = (rf, nn_model, scaler)
                print(f"done\n")
        
        self.trained = True
        print("All models trained successfully!\n")
    
    def predict(self, X, player_rank):
        """predict rank using the appropriate model based on player's actual rank"""
        if not self.trained:
            raise RuntimeError("need to train first!")
        
        if X is None or len(X) == 0:
            raise ValueError("empty input")
        
        if X.shape[1] != self.input_size:
            raise ValueError(f"expected {self.input_size} features, got {X.shape[1]}")
        
        # we can only handle 1-10k
        if player_rank < 1 or player_rank > 10000:
            raise ValueError(f"rank {player_rank} out of range (only 1-10000 supported)")
        
        range_key, _, mode = self._get_rank_range_key(player_rank)
        
        if mode == "log":
            models = self.models_log
        else:
            models = self.models_linear
        
        # fallback if we dont have this range trained
        if range_key not in models:
            range_key = list(models.keys())[-1]
        
        rf, nn_model, scaler = models[range_key]
        X_scaled = scaler.transform(X)
        
        # get both predictions
        rf_pred = rf.predict(X_scaled)
        
        nn_model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            nn_pred = nn_model(X_t).cpu().numpy().flatten()
        
        # blend em (RF is usually more stable)
        pred = 0.6 * rf_pred + 0.4 * nn_pred
        
        if mode == "log":
            pred = np.expm1(pred)  # undo log1p
        
        return np.clip(pred, 1, 10000)
    
    def save(self, path):
        """dump models to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {'log': {}, 'linear': {}}
        
        for rk, (rf, nn, sc) in self.models_log.items():
            data['log'][rk] = {'rf': rf, 'nn': nn.state_dict(), 'scaler': sc}
        
        for rk, (rf, nn, sc) in self.models_linear.items():
            data['linear'][rk] = {'rf': rf, 'nn': nn.state_dict(), 'scaler': sc}
        
        torch.save({'models': data, 'input_size': self.input_size}, path)
        print(f"saved to {path}")
    
    def load(self, path):
        """load models from disk"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no model at {path}")
        
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        for rk, d in ckpt['models']['log'].items():
            nn = SimpleNN(ckpt['input_size']).to(self.device)
            nn.load_state_dict(d['nn'])
            self.models_log[rk] = (d['rf'], nn, d['scaler'])
        
        for rk, d in ckpt['models']['linear'].items():
            nn = SimpleNN(ckpt['input_size']).to(self.device)
            nn.load_state_dict(d['nn'])
            self.models_linear[rk] = (d['rf'], nn, d['scaler'])
        
        self.trained = True
        print(f"loaded from {path}")
