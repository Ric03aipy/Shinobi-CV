# QUESTO NON è UN FILE DA MOSTRARE, è PER ME --> .GITIGNORE

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import ROOT, HAND_TRACKER_MODEL
from utility import draw_landmarks_on_image

# TODO: 
# PASSARE LA CAMERA DA WINDOWS A WSL
# APRIRE LA CAMERA
# HAND DETECTION
# DISPLAY DELLE MANI IN REAL TIME


img = cv.imread(ROOT.parent.parent / "hand_img.jpg")
cv.imshow("test_img", img)

# VISUAL TEST --- OK 
# cv.waitKey(0)   
# cv.destroyAllWindows()

# Create an HandLandmarker object
base_options = python.BaseOptions(model_asset_path = HAND_TRACKER_MODEL)
options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)

# Create the detector - context manager for correct destruction of the object 
with vision.HandLandmarker.create_from_options(options) as detector:
    image = mp.Image.create_from_file(str(ROOT.parent.parent / "hand_img.jpg"))
    detection_result = detector.detect(image)

    print(detection_result)
    
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # VISUAL TEST --- OK
    cv.imshow("titolo", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
    cv.waitKey(0)
    cv.destroyAllWindows()

