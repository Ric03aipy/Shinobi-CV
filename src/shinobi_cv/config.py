from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # src
DATA_PATH = ROOT / "data" 
MODELS_PATH = ROOT / "models"


HAND_TRACKER_MODEL = str(MODELS_PATH / "hand_landmarker.task")

INTERVAL = 5 # number of frames to discard
SIGNS = {idx:name for idx, name in enumerate(("dragon", "tiger", "dog", "rat", "ram", "horse", "monkey", "bird", "ox", "serpent", "hare", "boar"))}
SIGN_TO_REGISTER_ID = 3
DATA_FILE = DATA_PATH / "signs.csv" # label: points (21), hand, sign - to record for each hand detected

