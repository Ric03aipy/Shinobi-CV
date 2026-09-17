import time
import numpy as np
import joblib
import torch
import cv2 as cv
from collections import deque

from video_stramer import VideoCameraStreamer
from hand_tracking import HandDetector
from face_landmark_detection import FaceLandMarkDetection
from training.mlp_model import HandSignMLP
from animation import Renderer, TimedAnimation, ToggleAnimation, ANIMATION_PATH
from audio import AudioDetector
from utility import get_camera_coordinates

from config import  MODEL, MODELS_PATH, SIGNS, MINIMUM_CONFIDENCE, \
                    TIME_TO_CONFIRM, SEQUENCES, SEQUENCES_TO_CAST, MAX_QUEUE_LEN, \
                    SHARINGAN_MSG, BYAKUGAN_MSG, EMPTY_MSG
from sign_search import SignTrie

import threading
import queue


class Inference_Model(): 

    """Interface for inference regardless the model chosen.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        model_path = str(MODELS_PATH / model_name)
        match model_name: 
            case "rf_model": 
                self.model = joblib.load(model_path + ".pkl")
            case "mlp_model":
                self.model = HandSignMLP()
                weight_dict = torch.load(model_path + ".pth")
                self.model.load_state_dict(weight_dict)
            case "gnn_model":
                raise NotImplementedError("Still a #TODO. Try other models.")

    def inference(self, in_data:np.ndarray) -> tuple[int, float]:

        """Predict over input data.

        Args:
            in_data (np.ndarray): 1-D array from the hand detection. 

        Returns:
            int: class of the sign.
            float: confidence.
        """

        match self.model_name:
            case "rf_model": 
                # Predict
                probs = self.model.predict_proba(in_data.reshape(1, -1))
                pred = np.argmax(probs).astype(np.int32) # .item()
                confidence = np.max(probs) # .item()
            case "mlp_model" | "gnn_model":
                # Transform data in tensor
                in_data_t = torch.tensor(in_data, dtype=torch.float32)

                # Predict
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(in_data_t) 
                    probs = torch.softmax(logits, dim = 0)
                    confidence, pred = torch.max(probs, dim = 0) # max() in torch is max & argmax together 
            case _: 
                raise Exception("Anomaly in the model choice.") 

        return pred.item(), confidence.item() # Since numpy and torch both use .item() to extract the primitive datum



def live_inference(
        streamer: VideoCameraStreamer, 
        hand_detector: HandDetector, 
        model: Inference_Model, 
        sign_trie: SignTrie,
        face_detector: FaceLandMarkDetection,
        renderer: Renderer,
        msg_queue:queue.Queue
    ): 

    frame_streamer = streamer.stream_data()
    try: 
        frame = next(frame_streamer) # Initialize the first frame to elaborate
    except StopIteration: # Specific signal when an iterable is consumed completely
        print("Leaving at first iteration. Something wrong with the camera?")
        return 
    
    
    voting_queue = deque()  # For inference via majority vote
    votes = [0]*len(SIGNS)  # Tracks [sign_idx]=sign_frequency. Update this data structure instead of recomputing frequency at each step.

    prev_sign:int = None
    current_sign:int = None

    last_record = 0         # Instant of time when a new sign has been recorded

    # START = time.time()

    try:
        while True: # I can't stream directly from the generator because using next() and send() will consume 2 frames per iteration otherwise 

            # Hand & Face detector inference on the raw frame
            hand_detector_res, new_frame = hand_detector.detect(frame)
            face_detector_res = face_detector.detect_absolute_coordinates(frame, streamer.w, streamer.h)

            # (x, y) for each point + scale_factor info 
            core_pts = {}

            if face_detector_res: 
                core_pts["nose_tip"] = face_detector_res["nose_tip"];  core_pts["left_eye_center"] = face_detector_res["left_eye_center"];  core_pts["right_eye_center"] = face_detector_res["right_eye_center"]; core_pts["mouth_center"] = face_detector_res["mouth_center"]
            
            for hand in hand_detector_res:
                core_pts[f"hand_{hand['type']}"] = get_camera_coordinates((*hand["mean"], -1), streamer.w, streamer.h)

            # ====== Animation ======

            # Draw only specific points, not all detected pts 
            # TODO: modificare sopra la funzione hand_detector.detect() ? Molstrare i punti serve solo a me a capire come posizionarmi meglio durante l'uso. Per ora resta
            for pt in core_pts.values():
                pt_x, pt_y = pt[0], pt[1]
                # If the rectangle is not fully in the screen it doesn't raise any error
                new_frame = cv.rectangle(new_frame, (pt_x - 3, pt_y - 3), (pt_x + 3, pt_y + 3), (255, 255, 0), 2)

            # Reading from audio module
            try: 
                msg:str = msg_queue.get(block=False) # If no item is available raise queue.Empty exception, do not block
                if msg == SHARINGAN_MSG: 
                    if renderer.is_active("left_sharingan"):
                        renderer.end_animation("left_sharingan")
                    else: 
                        renderer.start_animation("left_sharingan")
                    if renderer.is_active("right_sharingan"):
                        renderer.end_animation("right_sharingan")  
                    else:
                        renderer.start_animation("right_sharingan")           
                elif msg == BYAKUGAN_MSG: 
                    if renderer.is_active("left_byakugan"):
                        renderer.end_animation("left_byakugan")
                    else: 
                        renderer.start_animation("left_byakugan")
                    if renderer.is_active("right_byakugan"):
                        renderer.end_animation("right_byakugan")  
                    else:
                        renderer.start_animation("right_byakugan")  
                else: 
                    print("Received garbage msg:", msg)
            except queue.Empty:
                pass
            except Exception as e:
                print(e)
                raise e
        


            # DEBUG
            # if time.time()-START > 4: 
            #     # animation_to_cast = "great fireball jutsu" 
            #     renderer.start_animation("great fireball jutsu")
            #     START = time.time()


            for animation_name in renderer.active_animations.copy(): 
                # Check to end animation 
                if not renderer.is_active(animation_name): 
                    renderer.end_animation(animation_name)
                # Animate
                else: 
                    new_frame = renderer.blend(animation_name, new_frame, core_pts, face_detector_res.get("scale_factor", None))


            # ====== Collect data as the sign predictive model expects ======  

            # There must be 2 hands (still must send a new frame otherwise infinite loop will raise)
            if len(hand_detector_res) != 2: 
                frame = frame_streamer.send(new_frame) # send() is a call to the generator itself so it receives the result of 'yield' keyword
                continue
            hand_type1 = hand_detector_res[0]["type"]
            hand_type2 = hand_detector_res[1]["type"]

            # 2 left hands or 2 right hands are not accepted (still must send a new frame otherwise infinite loop will raise)
            if hand_type1 == hand_type2: 
                frame = frame_streamer.send(new_frame) # send() is a call to the generator itself so it receives the result of 'yield' keyword
                continue
            data1 = hand_detector_res[0]["data"]
            data2 = hand_detector_res[1]["data"]

            # Force the order of data to be "left then right" - 
            if hand_type1 == "right": data1, data2 = data2, data1

            data = np.concatenate((data1, data2)) # shape = (126,)

            # ====== Inference ======
        
            # Time control of UX / detector fast insertion of signs 
            if time.time() - last_record < TIME_TO_CONFIRM: 
                frame = frame_streamer.send(new_frame) # send() is a call to the generator itself so it receives the result of 'yield' keyword
                continue
            last_record = time.time()

            pred_class, confidence = model.inference(data)

            control_append = confidence > MINIMUM_CONFIDENCE
            filled = len(voting_queue) == MAX_QUEUE_LEN
            
            # Reduce the size of the queue only if I know I can insert something later -> keep the queue to a fixed size for stability of the vote
            if filled and control_append:  
                vote_idx = voting_queue.popleft()
                votes[vote_idx] -= 1

            # Insertion can happen with the queue having any length provided that the confidence is high enough
            if control_append: 
                voting_queue.append(pred_class)
                votes[pred_class] += 1
            print(voting_queue)

            # Current sign is the mode of data in the queue. For stability wait for the queue to be filled
            if filled: 
                prev_sign = current_sign
                most_frequent_vote = np.argmax(votes).item()
                current_sign = most_frequent_vote if votes[most_frequent_vote] > MAX_QUEUE_LEN // 2 else current_sign   # not SIGNS[most_frequent_vote] because I want the index of this
                print(current_sign, "|", prev_sign)
                # Look up into the trie to cast the justu only when it changes
                if current_sign != prev_sign: 
                    sequence_to_cast = sign_trie.search_from_current_ptr(current_sign)
                    print("Trovato:", sequence_to_cast)
                    if sequence_to_cast: # After a sing is casted whatever arrives will restart the sequence thanks to the prefix-free property of SignTrie
                        sequence_to_cast_names = tuple((SIGNS[sign_id] for sign_id in sequence_to_cast))
                        print(SEQUENCES_TO_CAST[sequence_to_cast_names])

                        # Casting a jutsu - Graphical Overlay
                        renderer.start_animation(SEQUENCES_TO_CAST[sequence_to_cast_names])

            frame = frame_streamer.send(new_frame) # send() is a call to the generator itself so it receives the result of 'yield' keyword

    except StopIteration:
        print("User clicked 'q' to stop the stream. Closing the program.") 
        return
    except Exception as e:
        print("="*50)
        print("Unexpected error: ")
        print(e, e.__traceback__)
        print("="*50)

if __name__ == "__main__":

    # ======= Initialization =======

    # Streamer for camera frames
    streamer = VideoCameraStreamer()

    # Hand detector for normalized scale invariant points 
    hand_detector = HandDetector()

    # Model for sign inference
    model = Inference_Model(MODEL.stem)  

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
    audio_thread.start()

    # Start the inference
    live_inference(streamer, hand_detector, model, sign_trie, face_detector, renderer, msg_queue)