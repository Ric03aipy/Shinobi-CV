import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from config import HAND_TRACKER_MODEL, TRACKING_CONFIDENCE
from utility import draw_landmarks_on_image

from mediapipe.tasks.python.components.containers import NormalizedLandmark
from typing import List


class HandDetector: 

    """
    Class to detects and process landmarks from hands. Stream flat array of hand points, invariant to scale and normalized with respect to the wrist for each hand. 
    """

    def __init__(self):
        # Create and HandLandmarker object detector
        base_options = python.BaseOptions(model_asset_path = HAND_TRACKER_MODEL)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=2, 
            min_hand_detection_confidence=TRACKING_CONFIDENCE, # Threshold to detect the hand
            min_hand_presence_confidence=TRACKING_CONFIDENCE,  # Threshold to say the hand is still there
            min_tracking_confidence=TRACKING_CONFIDENCE        # Threshold for continuous tracking
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, raw_frame: np.ndarray) -> tuple[dict, np.ndarray]: 
        """
        Args:
            raw_frame (np.ndarray): image from camera
        Returns:
            tuple[dict, np.ndarray]: detection info "type", "data", "mean" for each hand, and the annotated frame
        """
        image = mp.Image(mp.ImageFormat.SRGB, raw_frame) # "Creates an Image object from a numpy ndarray."

        detection_result = self.detector.detect(image)    
        frame = draw_landmarks_on_image(image.numpy_view(), detection_result)

        # There can be 0, 1 or 2 hands: the loop automatically handles it. 
        extracted_hands = []
        for handed, hand_pts in zip(detection_result.handedness, detection_result.hand_landmarks): 
            hand_type = handed[0].category_name.lower() # "Left", "Right" -> "left", "right"
            # Preprocessing 
            precessed_data, mean_pt = self._preprocess_landmarks(hand_pts) 
            extracted_hands.append({
                "type": hand_type,
                "data": precessed_data,
                "mean": mean_pt # (x, y)
            })

        return (extracted_hands, frame)

    def _preprocess_landmarks(self, data: List[NormalizedLandmark]) -> tuple[np.ndarray, tuple[float, float]]: 
        """
        Normalize point coordinates in order to have the wrists in (0,0,0) and points invariant to scale.

        Args:
            data (List[NormalizedLandmark]): hand landmarks from a detection by mediapipe hand tracking detector. 
        
        Returns:
            np.ndarray: Flat (1D) array of coordinates (x_i,y_i,z_i) for i in {0,..., 21} - normalized to wrist and invariant to scale.  
            list[int, int, int]: mean point of the hand in coordinates returned by the detector - normalized in the image frame.
        """

        mean_x = mean_y = 0 
        
        # Normalization w.r.t. the wrist
        wrist = data[0]
        all_data = [] # [[X],[Y],[Z]]
        for pt in data: 
            mean_x += pt.x
            mean_y += pt.y

            all_data.append(pt.x - wrist.x)
            all_data.append(pt.y - wrist.y)
            all_data.append(pt.z - wrist.z)

        flat_data = np.array(all_data)
        
        # Scale invariance to enforce every number to be in [-1, 1]
        flat_data = flat_data / np.max(np.abs(flat_data))

        # Mean over the number of points 
        n_pts = len(data)   # 21 for each hand
        mean_pt = (mean_x / n_pts, mean_y / n_pts)

        return flat_data, mean_pt # flat.shape=(63,)
