import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from osu_api import OsuAPI
from features import engineer_features
from model import RankStratifiedEnsembleModel as EnsembleModel
from config import DEVICE, MODELS_DIR


def get_valid_player():
    """Keep asking for player until valid one is found"""
    try:
        api = OsuAPI()
    except Exception as e:
        print(f"Failed to authenticate with osu! API: {e}")
        return None
    
    while True:
        try:
            username = input("\nEnter osu! username (or 'quit' to exit): ").strip()
            
            if username.lower() == 'quit':
                print("Goodbye!")
                return None
            
            if not username:
                print("Username cannot be empty!")
                continue
            
            print(f"Checking if '{username}' exists...")
            user = api.get_user(username)
            print(f"Player found!")
            return user, api
            
        except Exception as e:
            print(f"Player '{username}' does not exist")
            print("   Please try another username\n")
            continue


def predict_player(user, api):
    """Make prediction for a player"""
    try:
        username = user['username']
        user_id = user['id']
        
        print(f"\nFetching scores for {username}...")
        
        best = api.get_scores(user_id, 50, 0)
        recent = api.get_recent(user_id, 50, 0)
        
        if not best and not recent:
            print(f"No scores found for {username}")
            return False
        
        print(f"Found {len(best)} best plays, {len(recent)} recent plays")
        
        player_data = {'user': user, 'best_scores': best, 'recent_scores': recent}
        
        # Extract features
        print("Engineering features...")
        features = engineer_features(player_data)
        
        if features is None or features.isna().all():
            print("Could not engineer features")
            return False
        
        # Get actual rank
        actual_rank = features['rank']
        
        if actual_rank == 0:
            print("Player has no rank yet")
            return False
        
        # CHECK IF RANK IS IN TRAINING RANGE (1-10k)
        if actual_rank < 1 or actual_rank > 10000:
            print(f"\nRANK CONSTRAINT VIOLATION")
            print(f"   This model only supports ranks 1-10,000")
            print(f"   Your rank: {int(actual_rank):,}")
            print(f"\n   Reason:")
            print(f"   The osu! API only provides top 10k leaderboard.")
            print(f"   Beyond rank 10k, data cannot be collected due to API rate limiting.")
            print(f"   Training data is limited to ranks 1-10,000.")
            print(f"\n   Cannot make reliable prediction.\n")
            return False
        
        X = features.drop('rank').values.reshape(1, -1)
        
        # Load model and predict
        print("Loading model...")
        model = EnsembleModel(X.shape[1], DEVICE)
        model_path = MODELS_DIR / 'model.pth'
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            print("   Run: python scripts/2_train_model.py")
            return False
        
        try:
            model.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
        print("Making prediction...")
        try:
            pred_rank = model.predict(X, actual_rank)[0]
        except ValueError as e:
            print(f"\n{e}\n")
            return False
        
        # Calculate error - Round before calculating
        pred_rank_rounded = round(pred_rank)
        difference = abs(pred_rank_rounded - actual_rank)
        error_pct = (difference / actual_rank) * 100
        pred_rank_log = np.log1p(pred_rank)
        actual_rank_log = np.log1p(actual_rank)
        log_error = abs(pred_rank_log - actual_rank_log)
        
        # Determine mode used
        if actual_rank <= 1000:
            mode = "LOG-SCALED"
        else:
            mode = "LINEAR"
        
        # Display results
        print(f"\n{'='*70}")
        print(f"PREDICTION FOR {username.upper()}")
        print(f"{'='*70}")
        print(f"Predicted Rank:  {pred_rank_rounded:>30,}")
        print(f"Actual Rank:     {int(actual_rank):>30,}")
        print(f"Difference:      {int(difference):>30,}")
        print(f"Error (Linear):  {error_pct:>30.2f}%")
        print(f"Error (Log):     {log_error:>30.4f}")
        print(f"Mode Used:       {mode:>30}")
        
        print(f"\nPerformance Stats:")
        print(f"   Best Plays:     {int(features['best_count']):>25}")
        print(f"   Pass Rate:      {features['pass_rate']*100:>25.1f}%")
        print(f"   SS Plays:       {int(features['ss_count']):>25}")
        print(f"   S Plays:        {int(features['s_count']):>25}")
        print(f"   A Plays:        {int(features['a_count']):>25}")
        
        print(f"\nPlayer Stats:")
        print(f"   PP:             {features['pp']:>25,.0f}")
        print(f"   Accuracy:       {features['accuracy']:>25.2f}%")
        print(f"   Play Count:     {int(features['play_count']):>25,}")
        print(f"   Level:          {int(features['level']):>25}")
        print(f"   PP Gain (month):{int(features['pp_gain_month']):>25}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("\n" + "="*70)
    print("osu! RANK PREDICTOR v3 (DUAL-MODE: LOG + LINEAR)")
    print("CONSTRAINT: Ranks 1-10,000 ONLY")
    print("   Reason: API rate limiting prevents data collection beyond top 10k")
    print("="*70)
    
    try:
        while True:
            # Get valid player
            result = get_valid_player()
            
            if result is None:
                break
            
            user, api = result
            
            # Make prediction
            success = predict_player(user, api)
            
            if success:
                # Ask to predict another
                again = input("🔄 Predict another player? (yes/no): ").strip().lower()
                if again not in ['yes', 'y']:
                    print("\n🙏 Thanks for using osu! Rank Predictor! 👋\n")
                    break
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
