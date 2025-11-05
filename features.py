import pandas as pd
import numpy as np
from datetime import datetime


def engineer_features(player_data):
    """Create advanced features from player data"""
    try:
        user = player_data['user']
        best = player_data['best_scores']
        recent = player_data['recent_scores']
        
        stats = user.get('statistics', {})
        
        features = {
            'rank': stats.get('global_rank', np.nan),
            'pp': stats.get('pp', 0),
            'accuracy': stats.get('hit_accuracy', 0),
            'play_count': stats.get('play_count', 0),
            'level': stats.get('level', {}).get('current', 0),
            'total_score': stats.get('total_score', 0),
        }
        
        # ===== BEST PLAYS FEATURES =====
        if best:
            pp_values = np.array([max(s.get('pp', 0) or 0, 0) for s in best[:50]])
            accuracy_values = np.array([s.get('accuracy', 0) or 0 for s in best[:20]])
            combo_values = np.array([s.get('max_combo', 0) or 0 for s in best[:20]])
            
            #Basic statistics
            features['best_pp_avg'] = np.mean(pp_values[:10]) if len(pp_values) > 0 else 0
            features['best_pp_median'] = np.median(pp_values) if len(pp_values) > 0 else 0
            features['best_accuracy_avg'] = np.mean(accuracy_values) if len(accuracy_values) > 0 else 0
            features['best_combo_avg'] = np.mean(combo_values) if len(combo_values) > 0 else 0
            features['best_count'] = len(best)
            
            #Pass plays analysis
            pass_plays = [s for s in best if s.get('passed', True)]
            features['pass_plays_count'] = len(pass_plays)
            features['pass_rate'] = len(pass_plays) / len(best) if len(best) > 0 else 0
            
            #Grade distribution
            grades = {'SS': 0, 'S': 0, 'A': 0, 'B': 0, 'C': 0}
            for score in pass_plays[:50]:
                rank = score.get('rank', 'N')
                #Check if rank is not None before upper()
                if rank is not None:
                    rank = rank.upper()
                    if rank in grades:
                        grades[rank] += 1
            
            features['ss_count'] = grades['SS']
            features['s_count'] = grades['S']
            features['a_count'] = grades['A']
            features['b_count'] = grades['B']
            features['c_count'] = grades['C']
            
            #Grade diversity
            total_grades = sum(grades.values())
            features['grade_diversity'] = len([g for g in grades.values() if g > 0]) / 5 if total_grades > 0 else 0
            
            #Peak performance
            features['peak_pp'] = max(pp_values) if len(pp_values) > 0 else 0
            features['peak_to_avg_ratio'] = features['peak_pp'] / max(features['best_pp_avg'], 1)
            
            #Consistency metrics
            features['pp_std'] = np.std(pp_values) if len(pp_values) > 1 else 0
            features['pp_range'] = (max(pp_values) - min(pp_values)) if len(pp_values) > 0 else 0
            features['consistency_score'] = 1 / (1 + features['pp_std'])  # 0-1 score
            
            #Accuracy consistency
            features['accuracy_std'] = np.std(accuracy_values) if len(accuracy_values) > 1 else 0
            features['min_accuracy'] = np.min(accuracy_values) if len(accuracy_values) > 0 else 0
            features['max_accuracy'] = np.max(accuracy_values) if len(accuracy_values) > 0 else 0
            
            #Combo analysis
            features['max_combo'] = np.max(combo_values) if len(combo_values) > 0 else 0
            features['avg_combo'] = np.mean(combo_values) if len(combo_values) > 0 else 0
            
            #Ranked score ratio
            features['ranked_score'] = stats.get('ranked_score', 0) or 0
            features['ranked_score_ratio'] = features['ranked_score'] / max(features['total_score'], 1)
            
        else:
            # Default values if no best plays
            features['best_pp_avg'] = 0
            features['best_pp_median'] = 0
            features['best_accuracy_avg'] = 0
            features['best_combo_avg'] = 0
            features['best_count'] = 0
            features['pass_plays_count'] = 0
            features['pass_rate'] = 0
            features['ss_count'] = 0
            features['s_count'] = 0
            features['a_count'] = 0
            features['b_count'] = 0
            features['c_count'] = 0
            features['grade_diversity'] = 0
            features['peak_pp'] = 0
            features['peak_to_avg_ratio'] = 0
            features['pp_std'] = 0
            features['pp_range'] = 0
            features['consistency_score'] = 0
            features['accuracy_std'] = 0
            features['min_accuracy'] = 0
            features['max_accuracy'] = 0
            features['max_combo'] = 0
            features['avg_combo'] = 0
            features['ranked_score'] = 0
            features['ranked_score_ratio'] = 0
        
        # ===== RECENT ACTIVITY FEATURES =====
        if recent:
            pass_recent = [s for s in recent if s.get('passed', True)]
            recent_pp_values = np.array([max(s.get('pp', 0) or 0, 0) for s in pass_recent[:50]])
            
            now = datetime.utcnow()
            pp_week = 0
            pp_month = 0
            pp_quarter = 0
            plays_week = 0
            plays_month = 0
            
            for score in pass_recent[:100]:
                try:
                    created_at = score.get('created_at')
                    if created_at is None:
                        continue
                    
                    score_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    days_ago = (now - score_date).days
                    pp = max(score.get('pp', 0) or 0, 0)
                    
                    if days_ago <= 7:
                        pp_week += pp
                        plays_week += 1
                    if days_ago <= 30:
                        pp_month += pp
                        plays_month += 1
                    if days_ago <= 90:
                        pp_quarter += pp
                except:
                    continue
            
            features['pp_gain_week'] = pp_week
            features['pp_gain_month'] = pp_month
            features['pp_gain_quarter'] = pp_quarter
            features['plays_week'] = plays_week
            features['plays_month'] = plays_month
            
            #Activity trend
            features['recent_count'] = len(recent)
            features['recent_pass_count'] = len(pass_recent)
            features['fail_rate'] = (len(recent) - len(pass_recent)) / max(len(recent), 1)
            features['avg_recent_pp'] = np.mean(recent_pp_values) if len(recent_pp_values) > 0 else 0
            
            #Improvement trend
            if len(pass_recent) >= 40:
                old_pp = np.mean([max(s.get('pp', 0) or 0, 0) for s in pass_recent[-20:]])
                new_pp = np.mean([max(s.get('pp', 0) or 0, 0) for s in pass_recent[:20]])
                features['improvement_trend'] = (new_pp - old_pp) / max(old_pp, 1)
            else:
                features['improvement_trend'] = 0
            
            #Activity regularity (days between plays)
            if len(pass_recent) >= 2:
                recent_dates = []
                for score in pass_recent[:50]:
                    try:
                        created_at = score.get('created_at')
                        if created_at is None:
                            continue
                        
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        recent_dates.append(date)
                    except:
                        continue
                
                if len(recent_dates) >= 2:
                    date_range = (max(recent_dates) - min(recent_dates)).days
                    features['activity_range_days'] = max(date_range, 1)
                    features['plays_per_day'] = len(recent_dates) / max(date_range, 1)
                    features['avg_days_between_plays'] = date_range / max(len(recent_dates) - 1, 1)
                else:
                    features['activity_range_days'] = 0
                    features['plays_per_day'] = 0
                    features['avg_days_between_plays'] = 0
            else:
                features['activity_range_days'] = 0
                features['plays_per_day'] = 0
                features['avg_days_between_plays'] = 0
            
        else:
            features['pp_gain_week'] = 0
            features['pp_gain_month'] = 0
            features['pp_gain_quarter'] = 0
            features['plays_week'] = 0
            features['plays_month'] = 0
            features['recent_count'] = 0
            features['recent_pass_count'] = 0
            features['fail_rate'] = 0
            features['avg_recent_pp'] = 0
            features['improvement_trend'] = 0
            features['activity_range_days'] = 0
            features['plays_per_day'] = 0
            features['avg_days_between_plays'] = 0
        
        # ===== DERIVED METRICS =====
        #Skill indicators
        features['total_hits'] = stats.get('total_hits', 0) or 0
        features['hits_per_play'] = features['total_hits'] / max(features['play_count'], 1)
        
        #PP progression indicator
        features['pp_per_play'] = features['pp'] / max(features['play_count'], 1)
        features['pp_per_level'] = features['pp'] / max(features['level'], 1)
        
        #Score efficiency
        features['avg_score_per_play'] = features['total_score'] / max(features['play_count'], 1)
        
        #Multi-metric consistency
        features['overall_consistency'] = (
            features['consistency_score'] * 0.4 +
            features['pass_rate'] * 0.3 +
            features['ranked_score_ratio'] * 0.3
        )
        
        return pd.Series(features)
        
    except Exception as e:
        print(f"Error engineering features: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_dataset(players_data):
    """Process all players into dataframe"""
    features_list = []
    failed_count = 0
    
    for idx, player in enumerate(players_data):
        try:
            features = engineer_features(player)
            if features is not None and not features.isna().all():
                features_list.append(features)
            else:
                failed_count += 1
        except Exception as e:
            print(f"Error processing player {idx}: {e}")
            failed_count += 1
            continue
    
    if not features_list:
        print("No valid players processed!")
        return None
    
    df = pd.DataFrame(features_list)
    df = df.dropna(subset=['rank']).fillna(0)
    
    print(f"Processed {len(df)} players with {len(df.columns)} features")
    print(f"Failed: {failed_count} players")
    
    # Show feature statistics
    print(f"\nFeature Statistics:")
    print(f"   Total Features: {len(df.columns)}")
    print(f"   Rank Range: {int(df['rank'].min()):,} - {int(df['rank'].max()):,}")
    print(f"   PP Range: {df['pp'].min():.0f} - {df['pp'].max():.0f}")
    print(f"   Accuracy Range: {df['accuracy'].min():.2f}% - {df['accuracy'].max():.2f}%")
    
    return df
