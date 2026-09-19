from dataclasses import dataclass
import random
import numpy as np


from src.shinobi_cv.sign_search import SignTrie
from src.shinobi_cv.hand_tracking import HandDetector
from src.shinobi_cv.live_inference import Inference_Model
from src.shinobi_cv.config import _MLP_MODEL, _RF_MODEL

# ==================================================================
# 
# Due to system configuration (ROS2 interference) (otherwise no need to clear PYTHONPATH before running):
# pytest run with: PYTHONPATH="" uv run pytest tests/generic_test.py
# 
# pytest requires functions to start with 'test_'
# 
# ==================================================================



def test_trie(): 
    SEQUENCES = {
        "great fireball jutsu": ["horse", "tiger"], # Serpent, Ram, Monkey, Boar, Horse, Tiger
        "summoning jutsu": ["dog", "ram"], # Boar, Dog, Bird, Monkey, Ram.
    }
    trie = SignTrie(SEQUENCES)
    fail_search_seq = [2, 5] # "dog", "horse"
    succ_search_seq = [5, 1] # "horse", "tiger"
    assert trie.search_from_current_ptr(fail_search_seq[0]) == []       # start of a sequence
    assert trie.search_from_current_ptr(fail_search_seq[1]) == []       # start of a sequence
    assert trie.search_from_current_ptr(succ_search_seq[0]) == []       # start of a sequence
    assert trie.search_from_current_ptr(succ_search_seq[1]) == [5, 1]   # found
    assert trie.search_from_current_ptr("serpent") is None              # not found

def test_hand_detection_normalization(): 
    hand_det = HandDetector()

    # Simulate normalized points from mediapipe with points having .x, .y, .z attributes
    @dataclass
    class FakePoint: 
        x: float
        y: float
        z: float

    n_points = 21
    pts = [FakePoint(random.random(), random.random(), random.random()) for _ in range(n_points)]
    res, mean = hand_det._preprocess_landmarks(pts)

    # test centering
    assert res.shape == (63,)   # expected a flat array
    assert res[0] == res[1] == res[2] == 0.0    # res[0,1,2] are wrist.x,y,z
    assert len(mean) == 2
    assert abs(mean[0] - sum([pt.x for pt in pts]) / n_points) < 1e-3
    assert abs(mean[1] - sum([pt.y for pt in pts]) / n_points) < 1e-3


def test_model_facade(): 
    model1 = Inference_Model(_RF_MODEL.stem)
    model2 = Inference_Model(_MLP_MODEL.stem)
    fake_data = np.random.random(size=(126,)) # 63 * 2 hands
    res1 = model1.inference(fake_data)
    res2 = model2.inference(fake_data)
    assert len(res1) == len(res2)
    pred1, conf1 = res1
    pred2, conf2 = res2
    assert type(pred1) == type(pred2) == int
    assert type(conf1) == type(conf2) == float

