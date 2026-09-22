import json
import logging
import queue
import time
from pathlib import Path

import pyaudio
from pynput.keyboard import Key, Listener
from rapidfuzz import fuzz
from vosk import KaldiRecognizer, Model

from src.shinobi_cv.config import (
    AUDIO_MODEL_PATH,
    BYAKUGAN_MSG,
    EMPTY_MSG,
    ENABLE_FUZZY_SEARCH,
    FUZZY_THRESHOLD,
    SHARINGAN_MSG,
)

logger = logging.getLogger(__name__)



class AudioDetector: 
    def __init__(self):
        if not Path.exists(AUDIO_MODEL_PATH):
            
            raise FileNotFoundError("Be sure the path for the audio model is correct.")

        # Initialize model for vocabulary recognition
        self.recognizer = KaldiRecognizer(
            Model(str(AUDIO_MODEL_PATH)),               # model
            16000,                                      # samplerate
            # json.dumps(AUDIO_VOCABULARY_WITH_UNK)       # vocabulary
        )

        # Flag to listen
        self.is_talking = False

        logger.info("Audio Recognizer ready.")


    def _on_press(self, key): 
        if key == Key.space: self.is_talking = True

    def _on_release(self, key): 
        if key == Key.space: self.is_talking = False

    def _process_recognised_text(self, text:str, msg_queue:queue.Queue):
        msg = EMPTY_MSG
        if ENABLE_FUZZY_SEARCH: 
            fuzzy_res_sharingan = max(
                fuzz.partial_ratio("sharingan", text), 
                fuzz.partial_ratio("写輪眼", text)
            )
            fuzzy_res_byakugan = max(
                fuzz.partial_ratio("byakugan", text),
                fuzz.partial_ratio("白眼", text)
            )
            if fuzzy_res_byakugan >= FUZZY_THRESHOLD and fuzzy_res_sharingan >= FUZZY_THRESHOLD: msg = SHARINGAN_MSG if fuzzy_res_sharingan >= fuzzy_res_byakugan else BYAKUGAN_MSG
            elif fuzzy_res_byakugan >= FUZZY_THRESHOLD: msg = BYAKUGAN_MSG
            elif fuzzy_res_sharingan >= FUZZY_THRESHOLD: msg = SHARINGAN_MSG
        
        else: 
            if "sharingan" in text or "写 輪 眼" in text or "写輪眼" in text: msg = SHARINGAN_MSG
            elif "byakugan" in text or "白眼" in text: msg = BYAKUGAN_MSG

        msg_queue.put(msg, block=True)

    def run(self, msg_queue:queue.Queue): 

        # Keyboard for "push-to-talk" approach 
        keyboard_listener = Listener(on_press=self._on_press, on_release=self._on_release)
        keyboard_listener.start() # Listener is a thread

        logger.info("Keyboard listerer is listening.")

        # Init microphone
        mic = pyaudio.PyAudio()
        audio_stream = mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=0,  # "pulse"
            frames_per_buffer=8192
        )

        logger.info("Microphone is ready to listen.")

        try:
            while True:

                # Start reading only when the start button is pressed
                if self.is_talking:   

                    if audio_stream.is_stopped():
                        audio_stream.start_stream()  
                        logger.info("Microphone is listening...")           

                    data = audio_stream.read(num_frames=4096, exception_on_overflow=False)

                    # When there is a puase the model analyses the audio
                    if self.recognizer.AcceptWaveform(data): 
                        result_json = json.loads(self.recognizer.Result())
                        text = result_json.get("text", "")
                        self._process_recognised_text(text, msg_queue)

                else: 
                    if audio_stream.is_active():
                        audio_stream.stop_stream()
                        logger.info("Not listening. Sending recording...")
                        # Clean any last words got before release
                        result_json = json.loads(self.recognizer.FinalResult())
                        self._process_recognised_text(result_json.get("text", ""), msg_queue)

                    # Don't occupy resources for seconds-like
                    time.sleep(0.05)
                  
        except Exception:
            logger.exception("Unpredicted exception raised in the audio module.")
        finally:
            # Cleanup
            keyboard_listener.stop()
            audio_stream.stop_stream()
            audio_stream.close()
            mic.terminate()
            logger.info("Audio&Mic resourse released.")
