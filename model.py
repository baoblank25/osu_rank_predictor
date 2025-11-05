import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import numpy as np
from pathlib import Path


class SimpleNN(nn.Module):
    """Simple neural network"""
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class RankStratifiedEnsembleModel:
    """
    Ensemble with DUAL SCALING MODE (RANKS 1-10K ONLY):
    - Ranks 1-1000: Use LOG-SCALING (better for top players)
    - Ranks 1000-10k: Use LINEAR scaling (better for mid-tier players)
    
    Constraints: Only predicts ranks 1-10,000
    Reason: osu! API only provides top 10k leaderboard due to rate limiting
    """
    def __init__(self, input_size, device):
        self.device = device
        self.input_size = input_size
        
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
                
                # Train NN
                nn_model = SimpleNN(self.input_size).to(self.device)
                optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
                criterion = torch.nn.MSELoss()
                
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                y_t = torch.FloatTensor(y_range_log).unsqueeze(1).to(self.device)
                y_mean = y_t.mean()
                y_std = y_t.std() + 1e-8
                y_normalized = (y_t - y_mean) / y_std
                
                #Show training progress
                print(f"    Training NN (lr=0.01, epochs={epochs})...")
                
                for epoch in range(epochs):
                    indices_t = torch.randperm(len(X_t))
                    epoch_loss = 0
                    
                    for i in range(0, len(X_t), batch_size):
                        idx = indices_t[i:i+batch_size]
                        X_batch = X_t[idx]
                        y_batch = y_normalized[idx]
                        
                        optimizer.zero_grad()
                        pred = nn_model(X_batch)
                        loss = criterion(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    #Print loss every 20 epochs
                    if (epoch + 1) % 20 == 0:
                        print(f"      Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
                
                self.models_log[range_key] = (rf, nn_model, scaler)
                print(f"Trained (log-scaled)\n")
        
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
                
                #Train NN
                nn_model = SimpleNN(self.input_size).to(self.device)
                optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
                criterion = torch.nn.MSELoss()
                
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                y_t = torch.FloatTensor(y_range).unsqueeze(1).to(self.device)
                y_mean = y_t.mean()
                y_std = y_t.std() + 1e-8
                y_normalized = (y_t - y_mean) / y_std
                
                print(f"    Training NN (lr=0.01, epochs={epochs})...")
                
                for epoch in range(epochs):
                    indices_t = torch.randperm(len(X_t))
                    epoch_loss = 0
                    
                    for i in range(0, len(X_t), batch_size):
                        idx = indices_t[i:i+batch_size]
                        X_batch = X_t[idx]
                        y_batch = y_normalized[idx]
                        
                        optimizer.zero_grad()
                        pred = nn_model(X_batch)
                        loss = criterion(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    if (epoch + 1) % 20 == 0:
                        print(f"      Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
                
                self.models_linear[range_key] = (rf, nn_model, scaler)
                print(f"Trained (linear)\n")
        
        self.trained = True
        print("All models trained successfully!\n")
    
    def predict(self, X, player_rank):
        """Make prediction using the right scaling mode based on rank"""
        try:
            if not self.trained:
                raise ValueError("Model not trained!")
        
            if X is None or len(X) == 0:
                raise ValueError("No input data!")
        
            if X.shape[1] != self.input_size:
                raise ValueError(f"Input size mismatch!")
            
            #Constraint check: Rank must be 1-10k
            if player_rank < 1 or player_rank > 10000:
                raise ValueError(
                    f"RANK CONSTRAINT VIOLATION\n"
                    f"   This model only supports ranks 1-10,000\n"
                    f"   Requested rank: {int(player_rank):,}\n\n"
                    f"   Reason: osu! API only provides top 10k leaderboard.\n"
                    f"   Beyond rank 10k, data cannot be collected due to API rate limiting.\n"
                    f"   Training data is limited to ranks 1-10,000."
                )
            
            #Get the right range and mode
            range_key, _, mode = self._get_rank_range_key(player_rank)
            
            #Select model set based on mode
            if mode == "log":
                models = self.models_log
                print(f"Using LOG-SCALED model for rank {int(player_rank):,}")
            else:
                models = self.models_linear
                print(f"Using LINEAR model for rank {int(player_rank):,}")
            
            if range_key not in models:
                #Fallback
                range_key = sorted(models.keys())[-1]
            
            rf, nn_model, scaler = models[range_key]
            X_scaled = scaler.transform(X)
            
            #Get predictions
            rf_pred = rf.predict(X_scaled)
            
            nn_model.eval()
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                nn_pred = nn_model(X_t).cpu().numpy().flatten()
            
            #Average predictions
            pred = (0.6 * rf_pred + 0.4 * nn_pred)
            
            #Convert if log mode was used
            if mode == "log":
                pred = np.expm1(pred)  #Convert from log scale
            
            return np.clip(pred, 1, 10000)  #Clip to 10k max
        
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
    
    def save(self, path):
        """Save all models"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        models_to_save = {
            'log': {},
            'linear': {}
        }
        
        for range_key, (rf, nn_model, scaler) in self.models_log.items():
            models_to_save['log'][range_key] = {
                'rf': rf,
                'nn': nn_model.state_dict(),
                'scaler': scaler,
            }
        
        for range_key, (rf, nn_model, scaler) in self.models_linear.items():
            models_to_save['linear'][range_key] = {
                'rf': rf,
                'nn': nn_model.state_dict(),
                'scaler': scaler,
            }
        
        torch.save({
            'models': models_to_save,
            'input_size': self.input_size,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load all models"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load log models
        for range_key, model_data in checkpoint['models']['log'].items():
            rf = model_data['rf']
            nn_model = SimpleNN(checkpoint['input_size']).to(self.device)
            nn_model.load_state_dict(model_data['nn'])
            scaler = model_data['scaler']
            self.models_log[range_key] = (rf, nn_model, scaler)
        
        # Load linear models
        for range_key, model_data in checkpoint['models']['linear'].items():
            rf = model_data['rf']
            nn_model = SimpleNN(checkpoint['input_size']).to(self.device)
            nn_model.load_state_dict(model_data['nn'])
            scaler = model_data['scaler']
            self.models_linear[range_key] = (rf, nn_model, scaler)
        
        self.trained = True
        print(f"Model loaded from {path}")
