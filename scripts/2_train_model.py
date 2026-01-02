import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import PROCESSED_DIR, MODELS_DIR, DEVICE, EPOCHS, BATCH_SIZE
from model import RankStratifiedEnsembleModel as EnsembleModel


def run_cross_validation(X, y, n_splits=5):
    """Run k-fold cross-validation to validate model performance"""
    print("=" * 70)
    print(f"RUNNING {n_splits}-FOLD CROSS-VALIDATION")
    print("=" * 70 + "\n")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_metrics = {'mae': [], 'rmse': [], 'r2': [], 'log_mae': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"Fold {fold}/{n_splits}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = EnsembleModel(X_train.shape[1], DEVICE)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        # Predict
        y_pred = np.array([model.predict(X_val[i:i+1], y_val[i])[0] for i in range(len(X_val))])
        
        # Metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        log_mae = mean_absolute_error(np.log1p(y_val), np.log1p(y_pred))
        
        cv_metrics['mae'].append(mae)
        cv_metrics['rmse'].append(rmse)
        cv_metrics['r2'].append(r2)
        cv_metrics['log_mae'].append(log_mae)
        
        print(f"   Fold {fold} - MAE: {mae:,.0f} | R²: {r2:.4f}\n")
    
    print("=" * 70)
    print("CROSS-VALIDATION RESULTS:")
    print("=" * 70)
    print(f"   MAE:      {np.mean(cv_metrics['mae']):>10,.0f} ± {np.std(cv_metrics['mae']):,.0f}")
    print(f"   RMSE:     {np.mean(cv_metrics['rmse']):>10,.0f} ± {np.std(cv_metrics['rmse']):,.0f}")
    print(f"   R²:       {np.mean(cv_metrics['r2']):>10.4f} ± {np.std(cv_metrics['r2']):.4f}")
    print(f"   Log MAE:  {np.mean(cv_metrics['log_mae']):>10.4f} ± {np.std(cv_metrics['log_mae']):.4f}")
    print("=" * 70 + "\n")
    
    return cv_metrics


if __name__ == "__main__":
    try:
        
        # Load data
        data_file = PROCESSED_DIR / 'features.csv'
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        
        print(f"Loaded {len(df)} players with {len(df.columns)} features\n")
        
        X = df.drop('rank', axis=1).values
        y = df['rank'].values
        
        # Filter out ranks beyond 10k
        valid_mask = (y >= 1) & (y <= 10000)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            print(f"WARNING: Found {invalid_count} players outside 1-10k range")
            print(f"   Filtering out invalid ranks...\n")
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Data Split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test:  {len(X_test)} samples\n")
        
        # Create and train model
        print(f"Using device: {DEVICE}")
        print(f"Creating DUAL-MODE model with {X_train.shape[1]} features\n")
        
        model = EnsembleModel(X_train.shape[1], DEVICE)
        
        print("="*70)
        print("STARTING DUAL-MODE TRAINING (LOG + LINEAR)")
        print("CONSTRAINT: Ranks 1-10,000 ONLY")
        print("   Reason: osu! API only provides top 10k leaderboard due to rate limiting")
        print("="*70 + "\n")
        
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        # Evaluate on test set
        print("\n" + "="*70)
        print("MAKING PREDICTIONS ON TEST SET")
        print("="*70 + "\n")
        
        # Make predictions (use actual rank for mode selection)
        y_pred = np.array([model.predict(X_test[i:i+1], y_test[i])[0] for i in range(len(X_test))])
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate log error
        y_test_log = np.log1p(y_test)
        y_pred_log = np.log1p(y_pred)
        log_mae = mean_absolute_error(y_test_log, y_pred_log)
        
        print(f"\n{'='*70}")
        print("TEST RESULTS")
        print(f"{'='*70}")
        print(f"   MAE (Linear):              {mae:>20,.0f}")
        print(f"   RMSE:                      {rmse:>20,.0f}")
        print(f"   MAE (Log Scale):           {log_mae:>20.4f}")
        print(f"   R² Score:                  {r2:>20.4f}")
        print(f"{'='*70}\n")
        
        # VARIANCE CHECK - Most Important!
        print("PREDICTION VARIANCE CHECK:")
        print(f"   Predicted values - Min: {y_pred.min():.0f}, Max: {y_pred.max():.0f}")
        print(f"   Predicted range: {y_pred.max() - y_pred.min():.0f}")
        print(f"   Predicted std dev: {y_pred.std():.0f}")
        print(f"   Actual range: {y_test.max() - y_test.min():.0f}")
        
        if y_pred.std() < 200:  # If predictions vary by less than 200 ranks
            print(f"\nWARNING: Model predictions have LOW variance!")
            print(f"   This suggests the model is predicting similar values for all inputs.")
            print(f"   Possible causes:")
            print(f"   1. Features aren't diverse enough")
            print(f"   2. Model hasn't learned proper patterns")
            print(f"   3. Need more training data\n")
        else:
            print(f"\nModel has good variance - predictions differ across inputs\n")
        
        # Show sample predictions
        print("Sample Predictions (showing diversity):")
        for i in range(min(5, len(X_test))):
            diff = abs(y_pred[i] - y_test[i])
            error_pct = (diff / y_test[i]) * 100
            print(f"   Actual: {int(y_test[i]):>8,} | Predicted: {int(y_pred[i]):>8,} | Error: {error_pct:>6.1f}%")
        print()
        
        model_path = MODELS_DIR / 'model.pth'
        model.save(model_path)
        
        print(f"Dual-mode model trained and saved!")
        print(f"Model supports ranks 1-10,000 ONLY")
        print(f"Ranks beyond 10k will be rejected during prediction")
        
        # Optional: Run cross-validation for model validation
        print("\n" + "=" * 70)
        print("OPTIONAL: Run cross-validation for comprehensive evaluation")
        print("=" * 70)
        print("   python scripts/4_evaluate_model.py")
        print("   This will run:")
        print("     - K-Fold Cross-Validation")
        print("     - Ablation Studies (feature importance)")
        print("     - Overfitting Detection")
        print("     - Learning Curve Analysis")
        
        print(f"\n Next step:")
        print(f"   python scripts/3_predict.py\n")
        
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
