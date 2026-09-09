import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.tasks.python.components.containers import NormalizedLandmark
from typing import List, Generator, Dict

from utility import draw_landmarks_on_image

import time
import numpy as np

from config import HAND_TRACKER_MODEL

class VideoCameraHandDetector: 

    def __init__(self, camera_index:int = 0):

        # Initialize the camera capture
        self.cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)
        if not self.cap.isOpened(): 
            raise ValueError("Cannot open the camera. Check for permissions, WSL binds and device availability.")

        # Required to make things work on WSL 
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

        # Original stream info
        self.w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS)) 

        print(f"Camera opened. Original strem info: width={self.w!s}, height={self.h!s}, fps={self.fps}")

        # TODO: TUTTE LE PRINT DI DEBUG O DI CHECK PUOI FARLE DIVENTARE LOG COSì IMPARI IL LOGGING + DECORATORS PER COMPATTARE IL CODICE DI LOGGING

        # Create and HandLandmarker object detector
        base_options = python.BaseOptions(model_asset_path = HAND_TRACKER_MODEL)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def stream_data(self) -> Generator[Dict, None, None]: 
        """
        Yields:
            Generator[Dict, None, None]: dictionary of handedness "type" and flattened hand points "data".
        """
        try:
            while True: 

                # Start time to display fps 
                start = time.time()

                # Read one frame
                ret, frame = self.cap.read()    # frame.shape = (480, 640, 3) | (H, W, C);  type(frame[0][0][0]) = numpy.uint8
                if not ret: 
                    print("Can't read further")
                    break
    
                # Create the detector - context manager for correct destruction of the object.
                # Also overwrite frame so that if this block fails frame still is the one captured from the camera.
                image = mp.Image(mp.ImageFormat.SRGB, frame) # "Creates an Image object from a numpy ndarray."

                detection_result = self.detector.detect(image)    
                frame = draw_landmarks_on_image(image.numpy_view(), detection_result)


                
                # There can be 0, 1 or 2 hands: the loop automatically handles it. 
                extracted_hands = []
                for handed, hand_pts in zip(detection_result.handedness, detection_result.hand_landmarks): 
                    hand_type = handed[0].category_name # "Left", "Right"
                    # Preprocessing
                    precessed_data = self._preprocess_landmarks(hand_pts) 
                    extracted_hands.append({
                        "type": hand_type,
                        "data": precessed_data
                    })

                # Write on the screen 
                cv.putText(frame, f"{self.fps!s} FPS", (50, 50), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0))

                # Compute fps = 1 / (end - start)
                self.fps = int(1 / (time.time() - start))

                # Display
                cv.imshow("Hand Tracker", frame)

                # Exit condition
                if cv.waitKey(1) == ord('q'): break

                # Generator Return 
                yield extracted_hands

        except Exception as e: 
            print(f"Exception caught: {e}")
        finally:
            # Cleanup (any case)
            self.cap.release()
            cv.destroyAllWindows()

    def _preprocess_landmarks(self, data: List[NormalizedLandmark]) -> np.ndarray: 
        """
        Normalize point coordinates in order to have the wrists in (0,0,0) and points invariant to scale.

        Args:
            data (List[NormalizedLandmark]): hand landmarks from a detection by mediapipe hand tracking detector. 
        
        Returns:
            np.ndarray: Flat (1D) array of coordinates (x_i,y_i,z_i) for i in {0,..., 21}.  
        """
        
        # Normalization w.r.t. the wrist
        wrist = data[0]
        all_data = [] # [[X],[Y],[Z]]
        for pt in data: 
            all_data.append(pt.x - wrist.x)
            all_data.append(pt.y - wrist.y)
            all_data.append(pt.z - wrist.z)

        flat_data = np.array(all_data)
        
        # Scale invariance to enforce every number to be in [-1, 1]
        flat_data = flat_data / np.max(np.abs(flat_data))
        return flat_data # shape=(63,)



# TODO: RIMUOVI QUESTE RIGHE
if __name__ == "__main__":
    v = VideoCameraHandDetector()
    v.run()