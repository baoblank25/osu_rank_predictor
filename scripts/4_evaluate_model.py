"""
Cross-validation script for the rank predictor

Runs k-fold CV to check if the model generalizes well or if its just
memorizing the training data. Also checks for overfitting by comparing
train vs validation error.

- Brian
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

from config import PROCESSED_DIR, MODELS_DIR, DEVICE, EPOCHS, BATCH_SIZE


def load_data():
    """Load the features csv, filter out any garbage ranks"""
    data_file = PROCESSED_DIR / 'features.csv'
    print(f"Loading data from {data_file}...")
    
    df = pd.read_csv(data_file)
    
    X = df.drop('rank', axis=1)
    y = df['rank'].values
    
    # only care about 1-10k since thats what the API gives us
    valid_mask = (y >= 1) & (y <= 10000)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Loaded {len(y)} valid players with {X.shape[1]} features\n")
    return X, y


def cross_validate(X, y, n_splits=5, random_state=42):
    """
    k-fold cross validation
    
    splits data into k parts, trains on k-1, tests on remaining
    repeats k times so every sample gets tested once
    """
    print("=" * 70)
    print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
    print("=" * 70 + "\n")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # metrics per fold
    fold_metrics = {
        'mae': [],
        'rmse': [],
        'r2': [],
        'log_mae': [],
        'train_mae': [],
    }
    
    X_array = X.values if hasattr(X, 'values') else X
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_array), 1):
        print(f"Fold {fold}/{n_splits}:")
        
        X_train, X_val = X_array[train_idx], X_array[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model (using Random Forest for consistency)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        # Clip predictions to valid range
        y_val_pred = np.clip(y_val_pred, 1, 10000)
        y_train_pred = np.clip(y_train_pred, 1, 10000)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)
        log_mae = mean_absolute_error(np.log1p(y_val), np.log1p(y_val_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        fold_metrics['mae'].append(mae)
        fold_metrics['rmse'].append(rmse)
        fold_metrics['r2'].append(r2)
        fold_metrics['log_mae'].append(log_mae)
        fold_metrics['train_mae'].append(train_mae)
        
        # Overfitting ratio
        overfit_ratio = train_mae / mae if mae > 0 else 0
        
        print(f"   Train samples: {len(y_train):,}, Val samples: {len(y_val):,}")
        print(f"   MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | R2: {r2:.4f}")
        print(f"   Train MAE: {train_mae:,.0f} | Overfit Ratio: {overfit_ratio:.3f}")
        print()
    
    # summary
    print("-" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("-" * 70)
    print(f"  MAE:     {np.mean(fold_metrics['mae']):,.0f} +/- {np.std(fold_metrics['mae']):,.0f}")
    print(f"  RMSE:    {np.mean(fold_metrics['rmse']):,.0f} +/- {np.std(fold_metrics['rmse']):,.0f}")
    print(f"  R2:      {np.mean(fold_metrics['r2']):.4f} +/- {np.std(fold_metrics['r2']):.4f}")
    print(f"  Log MAE: {np.mean(fold_metrics['log_mae']):.4f} +/- {np.std(fold_metrics['log_mae']):.4f}")
    print()
    
    # check for overfitting - if train error << val error, we have a problem
    avg_train_mae = np.mean(fold_metrics['train_mae'])
    avg_val_mae = np.mean(fold_metrics['mae'])
    gap = avg_val_mae - avg_train_mae
    ratio = avg_train_mae / avg_val_mae if avg_val_mae > 0 else 0
    
    print("OVERFITTING CHECK:")
    print(f"  Train MAE: {avg_train_mae:,.0f}")
    print(f"  Val MAE:   {avg_val_mae:,.0f}")
    print(f"  Gap: {gap:,.0f} (ratio: {ratio:.2f})")
    
    if ratio < 0.5:
        print("  >> heavy overfitting, model is memorizing training data")
    elif ratio < 0.7:
        print("  >> some overfitting, might want to add regularization")
    else:
        print("  >> looks fine, model generalizes ok")
    
    print()
    return fold_metrics


if __name__ == "__main__":
    try:
        # Load data
        X, y = load_data()
        
        # Run Cross-Validation
        cv_metrics = cross_validate(X, y, n_splits=5)
        
        # Final Summary
        print("EVALUATION COMPLETE")
        print(f"\nResults Summary:")
        print(f"   Cross-validation MAE: {np.mean(cv_metrics['mae']):,.0f} +/- {np.std(cv_metrics['mae']):,.0f}")
        print(f"   Cross-validation R2:  {np.mean(cv_metrics['r2']):.4f} +/- {np.std(cv_metrics['r2']):.4f}")
        
        print(f"\nRun this script after training to validate model performance.")
        print(f"   python scripts/4_evaluate_model.py\n")
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

