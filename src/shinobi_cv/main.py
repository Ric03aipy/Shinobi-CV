import logging

from src.shinobi_cv.live_inference import *

from src.shinobi_cv.animation import TimedAnimation, ToggleAnimation, ANIMATION_PATH
from src.shinobi_cv.audio import AudioDetector

import threading
from src.shinobi_cv.config import LOG_PATH, MODEL, SEQUENCES


if __name__ == "__main__":

    # ======= Logging setup ======

    logger = logging.getLogger()    # No name --> root logger: any logger created with name is a child of this one
    logger.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


    # ======= Initialization =======

    # Streamer for camera frames
    streamer = VideoCameraStreamer()

    # Hand detector for normalized scale invariant points 
    hand_detector = HandDetector()

    # Model for sign inference
    try:
        model = Inference_Model(MODEL.stem)  
    except Exception as e: 
        logger.critical("Model chosen not available.")
        raise

    # Trie for sequence search
    sign_trie = SignTrie(SEQUENCES)

    # Face Landmark Detector for mouth and eye normalized scale invariant points
    face_detector = FaceLandMarkDetection()

    # Animations
    fireball = TimedAnimation(3, ANIMATION_PATH / "flame_transparent.png", "mouth_center", particular_offset_x = -270)
    left_sharingan = ToggleAnimation(ANIMATION_PATH / "sharingan.png", "left_eye_center", particular_offset_x = -150, particular_offset_y = -150, particular_scale = 0.05)
    right_sharingan = ToggleAnimation(ANIMATION_PATH / "sharingan.png", "right_eye_center", particular_offset_x = -150, particular_offset_y = -150, particular_scale = 0.05)
    left_byakugan = ToggleAnimation(ANIMATION_PATH / "Byakugan.png", "left_eye_center", particular_offset_x = -200, particular_offset_y = -200, particular_scale=0.04)
    right_byakugan = ToggleAnimation(ANIMATION_PATH / "Byakugan.png", "right_eye_center", particular_offset_x = -200, particular_offset_y = -200, particular_scale=0.04)

    renderer = Renderer()
    renderer.add_animation("great fireball jutsu", fireball)
    renderer.add_animation("left_sharingan", left_sharingan)
    renderer.add_animation("right_sharingan", right_sharingan)
    renderer.add_animation("left_byakugan", left_byakugan)
    renderer.add_animation("right_byakugan", right_byakugan)

    # Audio module: init and start
    msg_queue = queue.Queue(maxsize=1)
    audio_detector = AudioDetector()
    audio_thread = threading.Thread(
        target=audio_detector.run,
        args=(msg_queue,),
        daemon=True
    )

    logger.info("All objects initialization terminated successfully.")

    audio_thread.start()

    logger.info("Thread audio listening. From now on space bar will enable push-to-talk feature.")
    logger.info("Staring app main loop.")

    # Start the inference
    live_inference(streamer, hand_detector, model, sign_trie, face_detector, renderer, msg_queue)