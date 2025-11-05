import os
from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv()

#Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'

#Create folders
for folder in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

#API Settings
OSU_CLIENT_ID = os.getenv('OSU_CLIENT_ID')
OSU_CLIENT_SECRET = os.getenv('OSU_CLIENT_SECRET')

#Training Settings
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001

#Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
