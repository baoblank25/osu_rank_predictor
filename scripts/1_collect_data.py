import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import requests
import time
import random
from features import process_dataset
from config import RAW_DIR, PROCESSED_DIR, OSU_CLIENT_ID, OSU_CLIENT_SECRET


class LeaderboardFetcher:
    """Fetch players from osu! global leaderboard (top 10,000 only)"""
    
    def __init__(self):
        self.base_url = 'https://osu.ppy.sh/api/v2'
        self.token = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with osu! API"""
        auth_url = 'https://osu.ppy.sh/oauth/token'
        data = {
            'client_id': OSU_CLIENT_ID,
            'client_secret': OSU_CLIENT_SECRET,
            'grant_type': 'client_credentials',
            'scope': 'public'
        }
        
        response = requests.post(auth_url, json=data, timeout=10)
        response.raise_for_status()
        self.token = response.json()['access_token']
        print("✓ Authenticated with osu! API")
    
    def _headers(self):
        return {'Authorization': f'Bearer {self.token}'}
    
    def get_page(self, page_num, retries=3):
        """Fetch a specific page with retry logic"""
        url = f'{self.base_url}/rankings/osu/performance'
        params = {'page': page_num}
        
        for attempt in range(retries):
            try:
                response = requests.get(
                    url, 
                    headers=self._headers(), 
                    params=params, 
                    timeout=15
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.Timeout:
                print(f"Timeout on page {page_num}, retry {attempt + 1}/{retries}")
                time.sleep(2 ** attempt)
            
            except requests.exceptions.ConnectionError:
                print(f"Connection error on page {page_num}, retry {attempt + 1}/{retries}")
                time.sleep(2 ** attempt)
            
            except Exception as e:
                print(f" Error on page {page_num}: {e}")
                time.sleep(1)
        
        return None
    
    def get_users_by_rank_randomized(self, rank_from, rank_to, count=10):
        """
        Fetch players within a rank range with RANDOMIZED selection
        Spreads across entire range instead of clustering at minimum
        """
        usernames = []
        
        if rank_from > 10000:
            print(f"Rank {rank_from:,} exceeds API limit (top 10,000 only)")
            return usernames
        
        # Adjust range to stay within 10,000
        rank_to = min(rank_to, 10000)
        
        print(f"Fetching {count} players from rank {rank_from:,} to {rank_to:,}...")
        print(f"Using RANDOMIZED distribution across range")
        
        # Calculate starting page (50 players per page)
        start_page = max(1, (rank_from - 1) // 50)
        end_page = (rank_to // 50) + 5
        
        # Randomize page order to avoid clustering
        all_pages = list(range(start_page, end_page))
        random.shuffle(all_pages)
        
        all_players_in_range = []  # Store all players found, then randomize
        
        try:
            for page in all_pages:
                data = self.get_page(page, retries=3)
                
                if data is None or 'ranking' not in data:
                    continue
                
                # Collect ALL players in range (not just first ones)
                for ranking in data['ranking']:
                    rank = ranking.get('global_rank')
                    username = ranking.get('user', {}).get('username')
                    
                    if rank and username and rank_from <= rank <= rank_to:
                        all_players_in_range.append((username, rank))
                
                time.sleep(0.3)
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        #RANDOMIZE SELECTION
        if len(all_players_in_range) > count:
            # Randomly select from entire pool
            selected = random.sample(all_players_in_range, count)
            usernames = [username for username, rank in selected]
            
            print(f"Randomly selected {count} from {len(all_players_in_range)} found players")
            
            # Sort by rank for display
            selected.sort(key=lambda x: x[1])
            for username, rank in selected:
                print(f"    ✓ {username} (rank #{rank:,})")
        else:
            usernames = [username for username, rank in all_players_in_range]
            print(f"  ✓ Found {len(usernames)} players")
            for username, rank in all_players_in_range:
                print(f"    ✓ {username} (rank #{rank:,})")
        
        print(f"  ✓ Fetched {len(usernames)} players")
        return usernames


def collect_by_rank_ranges():
    """
    Collect 80+ players from TOP 10,000 with RANDOMIZED distribution
    Cannot fetch beyond rank 10,000 due to API limitation
    """
    
    print("\n" + "="*70)
    print("COLLECTING 80+ PLAYERS FROM TOP 10,000 (RANDOMIZED)")
    print("API LIMITATION: Only top 10,000 players available")
    print("="*70 + "\n")
    
    fetcher = LeaderboardFetcher()
    all_usernames = []
    
    #ONLY RANK 1-10,000 (API LIMIT) with wider ranges for better randomization
    rank_ranges = [
        (1, 100, 10),              # 10 from rank 1-100
        (100, 500, 10),            # 10 from rank 100-500
        (500, 1000, 10),           # 10 from rank 500-1k
        (1000, 2000, 10),          # 10 from rank 1k-2k
        (2000, 4000, 10),          # 10 from rank 2k-4k
        (4000, 7000, 15),          # 15 from rank 4k-7k
        (7000, 10000, 15),         # 15 from rank 7k-10k
    ]
    
    print("📍 Target Rank Ranges (Top 10,000 Only - Randomized):")
    total_target = sum(count for _, _, count in rank_ranges)
    for rank_from, rank_to, count in rank_ranges:
        print(f"   • Rank {rank_from:,} - {rank_to:,}: {count} players")
    
    print(f"\nTotal target: {total_target} players")
    print("Randomized selection ensures diverse rank distribution\n")
    
    # Fetch players for each range
    for i, (rank_from, rank_to, count) in enumerate(rank_ranges, 1):
        print(f"\n{'='*70}")
        print(f"Range {i}/{len(rank_ranges)}: Rank {rank_from:,} - {rank_to:,}")
        print(f"{'='*70}")
        
        try:
            usernames = fetcher.get_users_by_rank_randomized(rank_from, rank_to, count)
            all_usernames.extend(usernames)
            
            total_so_far = len(all_usernames)
            percentage = (total_so_far / total_target) * 100
            
            print(f"\nRange complete: {len(usernames)} players")
            print(f"Total so far: {total_so_far}/{total_target} ({percentage:.1f}%)")
            
            if i < len(rank_ranges):
                print("Pausing 1 second before next range...")
                time.sleep(1)
        
        except Exception as e:
            print(f"Error in range: {e}")
    
    print(f"\n{'='*70}")
    print(f"Collection complete: {len(all_usernames)} total players")
    print(f"{'='*70}\n")
    
    return all_usernames


def collect_data(usernames):
    """Collect and process data for given usernames"""
    
    print(f"\n{'='*70}")
    print(f"COLLECTING DATA FOR {len(usernames)} PLAYERS")
    print(f"{'='*70}\n")
    
    from osu_api import OsuAPI
    
    try:
        api = OsuAPI()
        players_data = api.fetch_players(usernames)
        
        if not players_data:
            print("No player data collected!")
            return None
        
        print(f"\nSuccessfully collected data for {len(players_data)} players")
        
        # Save raw data
        raw_file = RAW_DIR / 'players.json'
        with open(raw_file, 'w') as f:
            json.dump(players_data, f, default=str, indent=2)
        print(f"Saved raw data to {raw_file}")
        
        # Process and save features
        df = process_dataset(players_data)
        
        if df is None or len(df) == 0:
            print("No valid features extracted!")
            return None
        
        processed_file = PROCESSED_DIR / 'features.csv'
        df.to_csv(processed_file, index=False)
        print(f"Saved features to {processed_file}")
        
        return df
    
    except Exception as e:
        print(f"Error collecting data: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_dataset(df):
    """Analyze dataset quality"""
    print(f"\n{'='*70}")
    print("DATASET ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nDataset Summary:")
    print(f"   Total Players: {len(df)}")
    print(f"   Total Features: {len(df.columns)}")
    print(f"   Rank Range: {int(df['rank'].min()):,} - {int(df['rank'].max()):,}")
    print(f"   Average Rank: {int(df['rank'].mean()):,}")
    print(f"   Median Rank: {int(df['rank'].median()):,}")
    
    print(f"\nPP Statistics:")
    print(f"   Min: {df['pp'].min():.0f} | Max: {df['pp'].max():.0f} | Avg: {df['pp'].mean():.0f}")
    
    print(f"\nAccuracy Statistics:")
    print(f"   Min: {df['accuracy'].min():.2f}% | Max: {df['accuracy'].max():.2f}% | Avg: {df['accuracy'].mean():.2f}%")
    
    print(f"\nData Quality:")
    missing = df.isnull().sum().sum()
    completeness = (1 - missing / (len(df) * len(df.columns))) * 100
    print(f"   Missing Values: {missing}")
    print(f"   Completeness: {completeness:.1f}%")
    
    print(f"\nRank Distribution (Better Diversity!):")
    quartiles = df['rank'].quantile([0.25, 0.5, 0.75])
    print(f"   Q1 (25%): {int(quartiles[0.25]):,}")
    print(f"   Q2 (50%): {int(quartiles[0.50]):,}")
    print(f"   Q3 (75%): {int(quartiles[0.75]):,}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    try:
        print("Starting data collection with RANDOMIZED distribution...\n")
        
        # Step 1: Collect usernames
        usernames = collect_by_rank_ranges()
        
        if not usernames or len(usernames) < 20:
            print("Not enough players collected!")
            print("Check your internet connection")
            exit(1)
        
        # Step 2: Fetch player data
        df = collect_data(usernames)
        
        if df is None or len(df) < 10:
            print("❌ Not enough valid data collected!")
            exit(1)
        
        # Step 3: Analyze dataset
        analyze_dataset(df)
        
        # Final message
        print("Data collection complete!")
        print("\nNext steps:")
        print("   python scripts/2_train_model.py")
        print("\nYour model is ready for training!\n")
        
    except KeyboardInterrupt:
        print("\n\nCollection stopped by user")
        exit(1)
    except Exception as e:
        print(f"\nCritical Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
