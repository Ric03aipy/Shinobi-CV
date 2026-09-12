from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # src
DATA_PATH = ROOT / "data" 
MODELS_PATH = ROOT / "models"


HAND_TRACKER_MODEL = str(MODELS_PATH / "hand_landmarker.task")
TRACKING_CONFIDENCE = 0.4
_RF_MODEL = MODELS_PATH / "rf_model.pkl"
_MLP_MODEL = MODELS_PATH / "mlp_model.pth"
_GNN_MODEL = ...    # TODO
MODEL = _RF_MODEL  # Change this to experiment with one of the models 

INTERVAL = 3 # Number of frames to discard
SIGNS = {idx:name for idx, name in enumerate(("dragon", "tiger", "dog", "rat", "ram", "horse", "monkey", "bird", "ox", "serpent", "hare", "boar"))}
MINIMUM_CONFIDENCE = 0.7
NUMBER_OF_FRAME_TO_CONFIRM = 5
SEQUENCES = {
    "great fireball jutsu": ("horse", "tiger"), # Serpent, Ram, Monkey, Boar, Horse, Tiger
    "summoning jutsu": ("dog", "ram"), # Boar, Dog, Bird, Monkey, Ram.
    "chidori": ("dragon", "rat") # Ox, Rabbit, Monkey, Dragon, Rat, Bird, Ox, Snake, Dog, Tiger, Monkey
}

SEQUENCES_TO_CAST = {t:s for s, t in SEQUENCES.items()}
