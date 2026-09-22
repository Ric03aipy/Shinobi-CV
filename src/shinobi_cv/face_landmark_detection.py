import logging

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark

from src.shinobi_cv.config import FACE_DETECTION_MODEL
from src.shinobi_cv.utility import get_camera_coordinates

logger = logging.getLogger(__name__)

class FaceLandMarkDetection: 

    def __init__(self):
        # Create an FaceLandmarker object
        base_options = python.BaseOptions(model_asset_path = FACE_DETECTION_MODEL)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        logger.info("Face model detector ready.")

    def detect_absolute_coordinates(self, raw_frame:np.ndarray, w: int, h:int) -> dict:
        """As detect, but point are returned in frame coordinates. Lost z info."""

        face_detector_res = self.detect(raw_frame)

        if face_detector_res:

            # Move to camera pixel coordinate. Receive only x and y for each point.
            face_detector_res["nose_tip"] = get_camera_coordinates(face_detector_res["nose_tip"], w, h)
            face_detector_res["left_eye_center"] = get_camera_coordinates(face_detector_res["left_eye_center"], w, h)
            face_detector_res["right_eye_center"] = get_camera_coordinates(face_detector_res["right_eye_center"], w, h)
            face_detector_res["mouth_center"] = get_camera_coordinates(face_detector_res["mouth_center"], w, h)

        return face_detector_res # [] if no face detected

    def detect(self, raw_frame: np.ndarray) -> dict: 
        """Run face detection on the input frame.
        Args:
            raw_frame (np.ndarray): image from camera

        Returns:
            dict: processed points, with keys "left_eye_center", "right_eye_center", "mouth_center", "scale_factor" for each element.
        """
        image = mp.Image(mp.ImageFormat.SRGB, raw_frame) # "Creates an Image object from a numpy ndarray."
        detection_result = self.detector.detect(image)  
        face_landmarks_lists = detection_result.face_landmarks

        # Assumption: only 1 face can be deteted - no configurable number of faces
        return self._preprocess_landmarks(face_landmarks_lists[0]) if face_landmarks_lists else {}

    def _preprocess_landmarks(self, data: list[NormalizedLandmark]) -> dict: 
        """
        Normalize point coordinates in order to have the nose tip in (0,0,0) and points invariant to scale.

        Args:
            data (List[NormalizedLandmark]): face landmarks from a detection by mediapipe face detector. 
        
        Returns: 
            dict: dictionary with points: "nose_tip", "left_eye_center", "right_eye_center", "mouth_center" and scale info "scale_factor". Each point is represented by 3 coordinates in a tuple (x,y,z).
        """

        # I use the map available at https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
        # to get info about specific points
        # A "stable distance": distance between extreme points of eyes is fairly invariant to facial expression. 
        # --> it can be the scale factor
        
        # Detect relevant point
        nose_tip = data[1] 
        left_eye_tip = data[33]
        left_iris_contour = (data[470], data[472], data[471], data[469])    # top, bottom, left, right
        right_eye_tip = data[263]
        right_iris_contour = (data[475], data[477], data[476], data[474])  # top, bottom, left, right
        mouth_contour = (data[0], data[17], data[61], data[291])           # top, bottom, left, right

        eye_dist = np.sqrt((left_eye_tip.x - right_eye_tip.x)**2 + (left_eye_tip.y - right_eye_tip.y)**2).item()

        # Preparing return values        
        
        left_eye_center = tuple((
            sum(pt.x for pt in left_iris_contour) / 4,
            sum(pt.y for pt in left_iris_contour) / 4,
            sum(pt.z for pt in left_iris_contour) / 4
        ))
        right_eye_center = tuple((
            sum(pt.x for pt in right_iris_contour) / 4,
            sum(pt.y for pt in right_iris_contour) / 4,
            sum(pt.z for pt in right_iris_contour) / 4
        ))
        mouth_center = tuple((
            sum(pt.x for pt in mouth_contour) / 4,
            sum(pt.y for pt in mouth_contour) / 4,
            sum(pt.z for pt in mouth_contour) / 4
        ))

        return {
            "nose_tip": (nose_tip.x, nose_tip.y, nose_tip.z),
            "left_eye_center": left_eye_center, 
            "right_eye_center": right_eye_center, 
            "mouth_center": mouth_center,
            "scale_factor": eye_dist 
        }
