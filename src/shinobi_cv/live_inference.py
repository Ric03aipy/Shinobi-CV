import numpy as np
import torch
import joblib


from collections import deque
from hand_tracking import VideoCameraHandDetector
from training.mlp_model import HandSignMLP

from config import MODEL, MODELS_PATH, SIGNS, MINIMUM_CONFIDENCE, NUMBER_OF_FRAME_TO_CONFIRM, SEQUENCES, SEQUENCES_TO_CAST
from sign_search import SignTrie


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


def live_inference(sign_trie:SignTrie): 

    # Initialize the hand detector and the iterable to get data on demand
    hand_detector = VideoCameraHandDetector()
    streamer = hand_detector.stream_data() 

    # Initialize the model
    model_name = MODEL.stem 
    model = Inference_Model(model_name)            

    # Initialize the queue for inference via majority vote
    voting_queue = deque() 
    max_len = 2 *  NUMBER_OF_FRAME_TO_CONFIRM # 51% of the que should agree at least --> 2 *
    votes = [0]*len(SIGNS) # Tracks [sign_idx]=sign_frequency. Update this data structure instead of recomputing frequency at each step.
    prev_sign = current_sign = None

    for stream_data in streamer: 

        # ====== Collect data as the model expects ======  

        # There must be 2 hands
        if len(stream_data) != 2: continue
        hand_type1 = stream_data[0]["type"]
        hand_type2 = stream_data[1]["type"]

        # 2 left hands or 2 right hands are not accepted   
        if hand_type1 == hand_type2: continue
        data1 = stream_data[0]["data"]
        data2 = stream_data[1]["data"]

        # Force the order of data to be "Left then Right" - 
        if hand_type1 == "Right": data1, data2 = data2, data1

        data = np.concatenate((data1, data2)) # shape = (126,)

        # ====== Inference ======
    
        pred_class, confidence = model.inference(data)

        control_append = confidence > MINIMUM_CONFIDENCE
        filled = len(voting_queue) == max_len
        
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
            current_sign = most_frequent_vote if votes[most_frequent_vote] > len(voting_queue) // 2 else current_sign   # not SIGNS[most_frequent_vote] because I want the index of this
            print(current_sign, "|", prev_sign)
            # Print === "Register the sign in the sequence to look into the trie to cast the justu" only when it changes
            if current_sign != prev_sign: 

                print("Entro nel ciclo con")
                # for registered_sign in sign_sequence: 
                sequence_to_cast = sign_trie.search_from_current_ptr(current_sign)
                print("Trovato:", sequence_to_cast)
                if sequence_to_cast: # After a sing is casted whatever arrives will restart the sequence thanks to the prefix-free property of SignTrie
                    sequence_to_cast_names = tuple((SIGNS[sign_id] for sign_id in sequence_to_cast))
                    print(SEQUENCES_TO_CAST[sequence_to_cast_names])

                if sequence_to_cast is None: 
                    print("Entor qui a causa di", current_sign)
                print("Esco dal ciclo")



if __name__ == "__main__":
    # Initialize the trie for sequence search
    sign_trie = SignTrie(SEQUENCES)
    # Start the inference
    live_inference(sign_trie)