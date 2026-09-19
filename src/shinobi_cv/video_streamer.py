import cv2 as cv
import numpy as np
import time

from typing import Generator

import logging
logger = logging.getLogger(__name__)

class VideoCameraStreamer: 
    """
    Class to open the camera. 
    """

    def __init__(self, camera_index:int = 0):

        # Initialize the camera capture
        self.cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)
        if not self.cap.isOpened(): 
            logger.critical("Cannot open the camera. Probably, the script can't access to it.")
            raise ValueError("Cannot open the camera. Check for permissions, WSL binds and device availability.")

        # Required to make things work on WSL 
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

        # Original stream info
        self.w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS)) 

        logger.info(f"Camera opened. Original strem info: width={self.w!s}, height={self.h!s}, fps={self.fps}")

    def stream_data(self) -> Generator[np.ndarray, np.ndarray, None]: # Yield_type, Send_type, Return_type
        """
        Yields:
            Generator[np.ndarray, None, None]: infinite generator of video frames. 
        """
        try:

            ret, frame = self.cap.read()    # frame.shape = (480, 640, 3) | (H, W, C);  type(frame[0][0][0]) = numpy.uint8
            if not ret: 
                print("Can't read further")
                return

            while True: 

                # Start time to display fps 
                start = time.time()

                # Generator Return - get from send updated frame with things drawn on it
                # The streamer sends a raw frame and waits for a generator.send(processed_frame) 
                processed_frame = yield frame

                # When the generator is called with next() - in a loop as an iterable for instance - processed_frame is None
                if processed_frame is not None: 
                    frame = processed_frame
                # else: the frame is the one read by the camera by default
    
                # Write on the screen the fps
                cv.putText(frame, f"{self.fps!s} FPS", (50, 50), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0))

                # Display
                cv.imshow("Hand Tracker", frame)

                # Exit condition
                if cv.waitKey(1) & 0xFF == ord('q'): 
                    logger.info("q pressed. Exiting the code correctly.")
                    break

                # Read one frame
                ret, frame = self.cap.read()    # frame.shape = (480, 640, 3) | (H, W, C);  type(frame[0][0][0]) = numpy.uint8
                if not ret: 
                    logger.warning("Can't read further.")
                    break 

                # Compute fps = 1 / (end - start)
                self.fps = int(1 / (time.time() - start))

        except Exception as e: 
            logger.exception("Unpredicted exception raised.")
        finally:
            # Cleanup (any case)
            self.cap.release()
            cv.destroyAllWindows()
            logger.info("All video resourser freed correctly.")

    