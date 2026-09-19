from src.shinobi_cv.config import SIGNS

import logging
logger = logging.getLogger(__name__)

class SignTrieNode: 
    def __init__(self):
        self.children = {}      
        self.formula = None

    def __repr__(self):
        return "[*] -> " + str(self.children)


# Instrad of a character I search a word and instead of a word I save a formula here

class SignTrie: 
    def __init__(self, sequences:dict=None):
        self.root = SignTrieNode()
        self.current_ptr = self.root

        # Initialize the trie
        sign_name_id_map = {name:idx for idx, name in SIGNS.items()}
        for seq in sequences: 
            self.insert([sign_name_id_map[word] for word in sequences[seq]])

        logger.info(f"Trie initialized with sequence: {str(sequences)}")

    def insert(self, formula: list[int]): 
        curr = self.root
        for id in formula: 
            if id not in curr.children:
                curr.children[id] = SignTrieNode() 
            curr = curr.children[id]
        curr.formula = formula
        # Enforce prefix-free code property: I assume that no jutsu formula can be part of another jutsu formula 
        # If == {} pass the assert. If there is any child raise AssertionError
        assert not curr.children, "Breaking prefix-free property; check sequences to put in the trie to not have subsequencies."
        

    def search_from_current_ptr(self, sign:int) -> list[int] | None:

        # Rematch against the root: sequence like ['tiger', 'tiger', 'dog', 'dragon'] is rejected at point ['tiger', 'tiger'], but it wouldn't recognise
        # [tiger', 'dog', 'dragon'] ; it's reasonable to not have equal signs next to each other, the last one should be counted as start of a sequence
        
        # If the pointer is not valid then reset to root
        if sign not in self.current_ptr.children: 
            self.current_ptr = self.root

        # Try 1 step if just reset to root; If not reset to root just test validity. In this case return something valid
        if sign in self.current_ptr.children: 
            self.current_ptr = self.current_ptr.children[sign]
            return self.current_ptr.formula if self.current_ptr.formula else [] # Can be [] or [smth] - to differenctiate from 'None'

        # 'None' is the value to say "the sign is not part of the current sequence neither it is the start of a new one"
        return None

    # per un controllo semplice questa print è ok, ma se il trie diventa grande sarebbe meglio un approccio più complesso stile dfs
    def __str__(self):
        return "root -> " + str(self.root.children)







# ========== TESTING ===============
# TODO: si può trasformare in testing professionale? servirebbe a tutto il codice, ma forse esula dagli obiettivi di questo progetto (in termini di CV? alla fine pytest è un insieme di assert a quanto ne so)


SEQUENCES = {
        "great fireball jutsu": ["horse", "tiger"], # Serpent, Ram, Monkey, Boar, Horse, Tiger
        "summoning jutsu": ["dog", "ram"], # Boar, Dog, Bird, Monkey, Ram.
}

if __name__ == "__main__": 

    trie = SignTrie(SEQUENCES)

    fail_search_seq = [2, 5] # "dog", "horse"
    succ_search_seq = [5, 1] # "horse", "tiger"

    # Simulate a stream
    for sign_id in fail_search_seq: 
        print("epslorando", sign_id)
        found = trie.search_from_current_ptr(sign_id)
        if found: 
            print("Parola trovata male!") # non deve essere stampato
            print(found)

    for sign_id in succ_search_seq: 
        print("epslorando", sign_id)
        found = trie.search_from_current_ptr(sign_id)
        if found: 
            print("Parola trovata bene!") # deve essere stampato
            print(found)
    for sign_id in fail_search_seq + [3] + succ_search_seq: # Assume that 
        print("epslorando", sign_id)
        found = trie.search_from_current_ptr(sign_id)
        if found: 
            print("Parola trovata mista!") # deve essere stampato
            print(found)



        