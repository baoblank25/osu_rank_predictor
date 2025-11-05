import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Error percentage
    rel_error = np.abs(y_true - y_pred) / y_true * 100
    within_5pct = (rel_error < 5).sum() / len(rel_error) * 100
    within_10pct = (rel_error < 10).sum() / len(rel_error) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'within_5pct': within_5pct,
        'within_10pct': within_10pct
    }

def print_metrics(metrics, title):
    """Print metrics nicely"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"{key.upper():<20} {value:>15.4f}")
    print(f"{'='*50}\n")
