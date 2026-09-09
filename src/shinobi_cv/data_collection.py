from hand_tracking import VideoCameraHandDetector

import numpy as np
import pandas as pd

from config import INTERVAL, SIGN_TO_REGISTER_ID, DATA_FILE, SIGNS
from pathlib import Path

COLUMN_NAME = [f"f_l_{i}" for i in range(126)]  + ["sign_id"]

def collect(): 

    hand_detector = VideoCameraHandDetector()
    count = 0
    print("Press 'q' to interrupt")
    streamer = hand_detector.stream_data()
    try: 

        # Ask the generator for data
        for stream_data in streamer: # It should be [{"type": "Left"|"Right", "data": np.ndarray}, {"type": "Left"|"Right", "data": np.ndarray}]

            # Consider a frame every INTERVAL
            count += 1
            if count != INTERVAL: continue
            count = 0

            # There must be 2 hands
            if len(stream_data) != 2: continue
            hand_type1 = stream_data[0]["type"]
            hand_type2 = stream_data[1]["type"]

            # 2 left hands or 2 right hands are not accepted   
            if hand_type1 == hand_type2: continue
            data1 = stream_data[0]["data"]
            data2 = stream_data[1]["data"]

            # Force the order of data to be "Left then Right" 
            if hand_type1 == "Right": 
                data1, data2 = data2, data1

            # Since hand type is fixed, there is no reason to write them as features since they would be the same for each row. Nothing to learn, just noise.
            # hand_type1, hand_type2 = np.array(0), np.array(1)

            # Encoding type and sign: Left/Right is binary 0/1, sign is categorial (0 to 11)
            # Create a single feature vector
            data = np.concatenate((data1, data2, np.array([SIGN_TO_REGISTER_ID]))) # shape = (127,)

        df = pd.DataFrame(data=[data], columns=COLUMN_NAME)
        df.to_csv(DATA_FILE, mode = 'a', index = 0, header = not Path.exists(DATA_FILE))
        print("Sample collected.")   

    except Exception: # 'q' stops the window and breaks - on purpose - the code 
        print("Data collection stopped.")
        
    finally:
        df = pd.read_csv(DATA_FILE)
        # print current status
        print(f"Total records for sign '{SIGNS[SIGN_TO_REGISTER_ID]} = {len(df[df['sign_id'] == SIGN_TO_REGISTER_ID])}")
        print(f"Total records = {len(df)}. Number of signs processed = {df['sign_id'].nunique()}/12")
        grouped = df.groupby(by="sign_id")
        print("Samples per sign: ")
        print(grouped["sign_id"].count().head(12))


if __name__ == "__main__":
    collect()