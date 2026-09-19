import time
from src.shinobi_cv.config import ROOT
from pathlib import Path
from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np

from typing import Literal

from src.shinobi_cv.config import SCALE_FACTOR_REFERENCE

ANIMATION_PATH = ROOT / "shinobi_cv" / "animations" 
MIN_SCALE = 0.005
MAX_SCALE = 2
LOCAL_DEBUG = False

import logging
logger = logging.getLogger(__name__)

# 2 tipes of animation: 
# TimedAnimation: animation with a fixed duration - for ninjustu
# ToggleAnimation: animation enabled and disabled with the same command - for dojutsu

class Animation(ABC): 

    # An abstact class can still have a concrete constructor
    def __init__(self, 
                 image_path:Path, 
                 roi:Literal["nose_tip", "mouth_center", "left_eye_center", "right_eye_center", "hand_left", "hand_right"], 
                 particular_offset_x:int=0, 
                 particular_offset_y:int=0,
                 particular_scale:float=1.0
                ):
        """
        Args:
            image_path (Path): where the animation is in the fyle system
            roi (Literal[&quot;nose_tip&quot;, &quot;mouth_center&quot;, &quot;left_eye_center&quot;, &quot;right_eye_center&quot;, &quot;hand_left&quot;, &quot;hand_right&quot;]): meaningful points
            particular_offset_x (int, optional): use this if you want to shift your image of the x axis. Defaults to 0
            particular_offset_y (int, optional): use this if you want to shift your image of the y axis. Defaults to 0.
            particular_scale (float, optional): use this if you want to resize your image in a custom way. Defaults to 1.0. 
        """
        if not image_path.exists(): 
            logger.critical("Could not proceed for missing file.")
            raise FileNotFoundError
        self.image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if self.image.shape[2] == 3:
            logger.warning(f"Animation at {image_path} has no alpha channel. Explicing it as solid. You should provide alpha channel.")
            self.image = cv.cvtColor(self.image, cv.COLOR_BGR2BGRA)
        self.roi = roi
        self.particular_offset_x = particular_offset_x
        self.particular_offset_y = particular_offset_y
        self.particular_scale = particular_scale

        # You must insert a reasonable offset
        h, w = self.image.shape[:2]
        if not (0 <= abs(self.particular_offset_x) <= w and 0 <= abs(self.particular_offset_y) <= h):
            raise ValueError("Invalid offset: it must be between 0 and maximum size of the image axis for each dimension.")
    
    @ abstractmethod
    def start(self): pass
    @ abstractmethod
    def is_active(self) -> bool: pass
    @ abstractmethod
    def get_image(self) -> np.ndarray: pass
    @ abstractmethod
    def end(self): pass
    @abstractmethod
    def get_offset_and_scale(self, core_pts:dict, scale:float) -> tuple[int|None, int|None, float]: 
        """
        Args:
            core_pts (dict): points where image can be attached.
            scale (float): stable distance provided by the face detector.

        Returns:
            tuple[int|None, int|None, float]: scaled offset and scaling for the image resize if computable.
        """
        pass



class TimedAnimation(Animation): 
    def __init__(self, 
                 duration: int | float, 
                 image_path:Path, 
                 roi:Literal["nose_tip", "mouth_center", "left_eye_center", "right_eye_center", "hand_left", "hand_right"], 
                 particular_offset_x:int=0, 
                 particular_offset_y:int=0,
                 particular_scale:float=1.0
                ):
        """
        Args:
            duration (int | float): duration of the animation.
            image_path (Path): where the animation is in the fyle system.
            roi (Literal[&quot;nose_tip&quot;, &quot;mouth_center&quot;, &quot;left_eye_center&quot;, &quot;right_eye_center&quot;, &quot;hand_left&quot;, &quot;hand_right&quot;]): meaningful points.
            particular_offset_x (int, optional): use this if you want to shift your image of the x axis. Defaults to 0.
            particular_offset_y (int, optional): use this if you want to shift your image of the y axis. Defaults to 0.
            particular_scale (float, optional): use this if you want to resize your image in a custom way. Defaults to 1.0. 
        """
        super().__init__(image_path, roi, particular_offset_x, particular_offset_y, particular_scale)
        self.duration = duration
        self.start_time = 0
        
    def start(self): self.start_time = time.time()

    def is_active(self) -> bool: return (time.time() - self.start_time) < self.duration

    def end(self): self.start_time = 0 

    def get_image(self) -> np.ndarray: return self.image

    def get_offset_and_scale(self, core_pts:dict[str, tuple[int, int]], scale:float) -> tuple[int|None, int|None, float]: 
        offset_x, offset_y = core_pts.get(self.roi, (None, None))  # Enters already only (x, y)
        if offset_x is None and offset_y is None: return (offset_x, offset_y, self.particular_scale)

        # Computing total scale
        if scale is None: scale = SCALE_FACTOR_REFERENCE
        scale = scale / SCALE_FACTOR_REFERENCE # > 1 if nearer -> bigger image

        total_scale = scale * self.particular_scale
        total_scale = max(MIN_SCALE, min(total_scale, MAX_SCALE))

        # Return scaled offset and scaling for the image resize
        
        return (offset_x + int(self.particular_offset_x * total_scale), 
                offset_y + int(self.particular_offset_y * total_scale), 
                total_scale)



class ToggleAnimation(Animation): 
    def __init__(self, 
                 image_path: Path, 
                 roi:Literal["nose_tip", "mouth_center", "left_eye_center", "right_eye_center", "hand_left", "hand_right"], 
                 particular_offset_x:int=0, 
                 particular_offset_y:int=0,
                 particular_scale:float=1.0
                ):
        super().__init__(image_path, roi, particular_offset_x, particular_offset_y, particular_scale)
        self.active = False

    def start(self): self.active = True

    def end(self): self.active = False

    def is_active(self) -> bool: return self.active    

    def get_image(self) -> np.ndarray: return self.image

    def get_offset_and_scale(self, core_pts:dict[str, tuple[int, int]], scale:float) -> tuple[int|None, int|None, float]: 
        offset_x, offset_y = core_pts.get(self.roi, (None, None))  # Enters already only (x, y)
        if offset_x is None and offset_y is None: return (offset_x, offset_y, self.particular_scale)
        if scale is None: scale = SCALE_FACTOR_REFERENCE
        scale = scale / SCALE_FACTOR_REFERENCE # > 1 if nearer -> bigger image

        total_scale = scale * self.particular_scale
        total_scale = max(MIN_SCALE, min(total_scale, MAX_SCALE))

        return (offset_x + int(self.particular_offset_x * total_scale), 
                offset_y + int(self.particular_offset_y * total_scale), 
                total_scale)


        
class Renderer: 

    # Questa classe deve gestire entrambe le tipologie di animazione
    
    def __init__(self):
        self.animations = dict()
        self.active_animations = set()

    def add_animation(self, animation_name:str, animation:Animation): 
        if animation_name in self.animations:
            raise ValueError("Animations must have unique names.")
        self.animations[animation_name] = animation

    def start_animation(self, animation_name:str): 
        self.active_animations.add(animation_name)
        self.animations[animation_name].start()

    def end_animation(self, animation_name:str): 
        self.animations[animation_name].end()
        self.active_animations.remove(animation_name)

    def is_active(self, animation_name:str) -> bool: return self.animations[animation_name].is_active()

    def get_animation(self, animation_name:str): return self.animations[animation_name]

    def blend(self, animation_name:str, background: np.ndarray, core_pts:dict, scale_factor:float|None) -> np.ndarray:
        """Place subject image on background handling proportion/shape difference and closeness to the camera."""

        offset_x, offset_y, scale = self.animations[animation_name].get_offset_and_scale(core_pts, scale_factor)
        subject = self.animations[animation_name].get_image()
        subject = cv.resize(subject, None, fx=scale, fy=scale)

        if offset_x is None and offset_y is None: 
            # No offset means there are no points to anchor
            return background
        
        subject_h, subject_w = subject.shape[:2]

        if LOCAL_DEBUG: logger.debug("forma del soggetto:", subject.shape)

        # Computing how many pixels are out of bound due to particular offset of images on the negative axis (the positive values are handled by slicing)
        x_oob = -min(0, offset_x)
        y_oob = -min(0, offset_y)

        # Extract only the portion of the background that superposes: slicing stops at bounds when out of image bounds.
        # For the background, ignore the transparency, if any
        # This robust selection of the roi_background makes some flickering around the border (top and left) but prevents from unpredictable negative index -> broadcast error
        roi_background = background[max(0, offset_y + y_oob): max(0, offset_y + subject_h), max(0, offset_x + x_oob): max(0, offset_x + subject_w)][:,:,:3]

        # If subject is larger than the canvas I have to cat the subject to the available space
        roi_subject = subject[y_oob: y_oob + roi_background.shape[0], x_oob: x_oob + roi_background.shape[1]]

        # Blending : note that height and width 
        alpha = roi_subject[:,:,3].astype(float) / 255.0 # Normalize in [0,1] ; 2D, shape es. for "great fireball jutsu" = (302, 290) 
        roi_subject_colors = roi_subject[:,:,:3].astype(float)
        if LOCAL_DEBUG: logger.debug("dimensione di alpha", alpha.shape)
        if LOCAL_DEBUG: logger.debug("dimensione di roi_sub_colors", roi_subject_colors.shape)
        if LOCAL_DEBUG: logger.debug("dimensione di roi_bg", roi_background.shape)
        blend_roi = (roi_subject_colors * alpha[..., np.newaxis] + roi_background * (1 - alpha[..., np.newaxis])).astype(np.uint8) # * is element-wise multiplication | adding a new axis to allow numpy broadcasting es. (302, 290, 1) with (302, 290, 3)

        # Glue the blend portion on the background and return 
        img_blend = background.copy()
        if LOCAL_DEBUG: logger.debug("dimensione di quello che voglio prendere:", img_blend[y_oob + offset_y: y_oob + offset_y + subject_h, x_oob + offset_x: x_oob + offset_x + subject_w].shape)
        if LOCAL_DEBUG: logger.debug("dimensione di quello che voglio assegnare:", blend_roi.shape)
        img_blend[y_oob + offset_y: y_oob + offset_y + roi_subject_colors.shape[0], x_oob + offset_x: x_oob + offset_x + roi_subject_colors.shape[1]] = blend_roi

        return img_blend

    def __getitem__(self, key): return self.get_animation(key)



