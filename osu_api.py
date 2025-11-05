import requests
import time
from config import OSU_CLIENT_ID, OSU_CLIENT_SECRET
from tqdm import tqdm

class OsuAPI:
    def __init__(self):
        self.base_url = 'https://osu.ppy.sh/api/v2'
        self.token = None
        self._authenticate()
    
    def _authenticate(self):
        """Get API token"""
        auth_url = 'https://osu.ppy.sh/oauth/token'
        data = {
            'client_id': OSU_CLIENT_ID,
            'client_secret': OSU_CLIENT_SECRET,
            'grant_type': 'client_credentials',
            'scope': 'public'
        }
        
        response = requests.post(auth_url, json=data)
        self.token = response.json()['access_token']
        print("Authenticated with osu! API")
    
    def _headers(self):
        return {'Authorization': f'Bearer {self.token}'}
    
    def get_user(self, username):
        url = f'{self.base_url}/users/{username}/osu'
    
        try:
            response = requests.get(url, headers=self._headers(), timeout=10)
            response.raise_for_status()
            data = response.json()
        
        #Check if player has valid data
            if not data or 'id' not in data:
                raise Exception(f"Invalid player data for {username}")
        
            return data
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise Exception(f"Player '{username}' not found")
            else:
                raise Exception(f"API Error: {response.status_code}")
    
        except requests.exceptions.Timeout:
            raise Exception("Request timed out")
    
        except Exception as e:
            raise Exception(f"Error fetching player: {str(e)}")
    
    def get_scores(self, user_id, limit=100, offset=0):
        """Get player's best scores"""
        url = f'{self.base_url}/users/{user_id}/scores/best'
        params = {'limit': min(limit, 100), 'offset': offset}
        response = requests.get(url, headers=self._headers(), params=params)
        return response.json()
    
    def get_recent(self, user_id, limit=100, offset=0):
        """Get player's recent scores"""
        url = f'{self.base_url}/users/{user_id}/scores/recent'
        params = {'limit': min(limit, 100), 'offset': offset}
        response = requests.get(url, headers=self._headers(), params=params)
        return response.json()
    
    def fetch_players(self, usernames):
        """Fetch data for multiple players"""
        players_data = []
        
        for username in tqdm(usernames, desc="Fetching players"):
            try:
                user = self.get_user(username)
                user_id = user['id']
                
                #Get best scores
                best_scores = []
                for offset in range(0, 100, 50):
                    scores = self.get_scores(user_id, 50, offset)
                    if not scores:
                        break
                    best_scores.extend(scores)
                
                #Get recent scores
                recent_scores = []
                for offset in range(0, 100, 50):
                    scores = self.get_recent(user_id, 50, offset)
                    if not scores:
                        break
                    recent_scores.extend(scores)
                
                players_data.append({
                    'user': user,
                    'best_scores': best_scores,
                    'recent_scores': recent_scores
                })
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching {username}: {e}")
        
        return players_data
