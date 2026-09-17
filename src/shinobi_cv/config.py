from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # src
DATA_PATH = ROOT / "data" 
MODELS_PATH = ROOT / "models"


HAND_TRACKER_MODEL = str(MODELS_PATH / "hand_landmarker.task")
FACE_DETECTION_MODEL = str(MODELS_PATH / "face_landmarker_v2_with_blendshapes.task")
TRACKING_CONFIDENCE = 0.4

_RF_MODEL = MODELS_PATH / "rf_model.pkl"
_MLP_MODEL = MODELS_PATH / "mlp_model.pth"
_GNN_MODEL = ...    # TODO
MODEL = _RF_MODEL  # Change this to experiment with one of the models 

_JAP_AUDIO_MODEL_PATH = Path(MODELS_PATH / "vosk-model-small-ja-0.22")
_ENG_AUDIO_MODEL_PATH = Path(MODELS_PATH / "vosk-model-small-en-us-0.15")
_ITA_AUDIO_MODEL_PATH = Path(MODELS_PATH / "vosk-model-small-it-0.22")
AUDIO_MODEL_PATH = _ITA_AUDIO_MODEL_PATH

SIGNS = {idx:name for idx, name in enumerate(("dragon", "tiger", "dog", "rat", "ram", "horse", "monkey", "bird", "ox", "serpent", "hare", "boar"))}
MINIMUM_CONFIDENCE = 0.7
TIME_TO_CONFIRM = 0.2  
MAX_QUEUE_LEN = 10
SEQUENCES = {
    "great fireball jutsu": ("horse", "rat"), # Serpent, Ram, Monkey, Boar, Horse, Tiger
    "summoning jutsu": ("dog", "ram"), # Boar, Dog, Bird, Monkey, Ram.
    "chidori": ("dragon", "rat") # Ox, Rabbit, Monkey, Dragon, Rat, Bird, Ox, Snake, Dog, Tiger, Monkey
}

SEQUENCES_TO_CAST = {t:s for s, t in SEQUENCES.items()}

SCALE_FACTOR_REFERENCE = 0.159

AUDIO_VOCABULARY_WITH_UNK = ["写輪眼", "白眼", "[unk]"]

SHARINGAN_MSG = "sharingan"
BYAKUGAN_MSG = "byakugan"
EMPTY_MSG = "_"
ENABLE_FUZZY_SEARCH = True
FUZZY_THRESHOLD = 60
