# TODO: RIALLINEARE RISPETTO AL REFACTORING FATTO SUIGLI OGGETTI INTERESSATI


from src.shinobi_cv.hand_tracking import HandDetector
from src.shinobi_cv.video_streamer import VideoCameraStreamer

import numpy as np
import pandas as pd

from src.shinobi_cv.config import DATA_FILE, SIGNS
from pathlib import Path

INTERVAL = 3 # Number of frames to discard

COLUMN_NAME = [f"f_l_{i}" for i in range(126)]  + ["sign_id"]
SIGN_TO_REGISTER_ID = 11 # TODO : devo fare monkey = 6

def collect(): 

    count = 0
    written = 0
    streamer = VideoCameraStreamer()
    hand_detector = HandDetector()
    frame_streamer = streamer.stream_data()

    print("Press 'q' to interrupt")

    try: 
        frame = next(frame_streamer) # Initialize the first frame to elaborate
    except StopIteration: # Specific signal when an iterable is consumed completely
        print("Leaving at first iteration. Something wrong with the camera?")
        return 

    try: 

        # Ask the generator for data
        while True: 

            hand_detector_res, new_frame = hand_detector.detect(frame)

            # There must be 2 hands
            if len(hand_detector_res) != 2: 
                frame = frame_streamer.send(new_frame) # send() is a call to the generator itself so it receives the result of 'yield' keyword
                continue
            hand_type1 = hand_detector_res[0]["type"]
            hand_type2 = hand_detector_res[1]["type"]

            # 2 left hands or 2 right hands are not accepted   
            if hand_type1 == hand_type2: 
                frame = frame_streamer.send(new_frame) # send() is a call to the generator itself so it receives the result of 'yield' keyword
                continue
            data1 = hand_detector_res[0]["data"]
            data2 = hand_detector_res[1]["data"]

            # Force the order of data to be "Left then Right" 
            if hand_type1 == "right": data1, data2 = data2, data1

            # Since hand type is fixed, there is no reason to write them as features since they would be the same for each row. Nothing to learn, just noise.
            # hand_type1, hand_type2 = np.array(0), np.array(1)

            # Consider a frame every INTERVAL
            count += 1
            if count != INTERVAL:
                frame = frame_streamer.send(new_frame) # send() is a call to the generator itself so it receives the result of 'yield' keyword
                continue
            count = 0   

            # Encoding type and sign: Left/Right is binary 0/1, sign is categorial (0 to 11)
            # Create a single feature vector
            collected_data = np.concatenate((data1, data2, np.array([SIGN_TO_REGISTER_ID]))) # shape = (127,)

            # Not the most efficient way to write but for this application it speeds up the boring process: no risk of no writing
            df = pd.DataFrame(data=[collected_data], columns=COLUMN_NAME)
            df.to_csv(DATA_FILE, mode = 'a', index = 0, header = not Path.exists(DATA_FILE))
            written += 1
            print(f"Sample collected in this session #{written}.")

            frame = frame_streamer.send(new_frame) # send() is a call to the generator itself so it receives the result of 'yield' keyword

    except Exception: # 'q' stops the window and breaks - on purpose - the code 
        print("Data collection stopped.")
        
    finally:
        # Print current status
        df = pd.read_csv(DATA_FILE)
        print(f"Total records for sign '{SIGNS[SIGN_TO_REGISTER_ID]}' = {len(df[df['sign_id'] == SIGN_TO_REGISTER_ID])}")
        print(f"Total records = {len(df)}. Number of signs processed = {df['sign_id'].nunique()}/12")
        grouped = df.groupby(by="sign_id")
        print("Samples per sign: ")
        print(grouped["sign_id"].count().head(12))


if __name__ == "__main__":
    collect()