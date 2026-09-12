import time
from config import ROOT
from pathlib import Path
from abc import ABC, abstractmethod

ANIMATION_PATH = ROOT / "shinobi_cv" / "animations" 

# 2 tipes of animation: 
# TimedAnimation: animation with a fixed duration - for ninjustu
# ToggleAnimation: animation enabled and disabled with the same command - for dojutsu

class Animation(ABC): 
    @ abstractmethod
    def start(self): pass
    @ abstractmethod
    def is_active(self) -> bool: pass
    @ abstractmethod
    def get_image_path(self) -> Path: pass
    @ abstractmethod
    def end(self): pass



class TimedAnimation(Animation): 
    def __init__(self, duration: int | float, image_path:Path):
        self.duration = duration
        self.start_time = 0
        self.image_path = image_path

    def start(self): self.start_time = time.time()

    def is_active(self) -> bool: return (time.time() - self.start_time) < self.duration

    def end(self): self.start_time = 0 

    def get_image_path(self) -> Path: return self.image_path



class ToggleAnimation(Animation): 
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.toggle = False

    def start(self): self.toggle = True

    def end(self): self.toggle = False

    def is_active(self) -> bool: return self.toggle    

    def get_image_path(self) -> Path: return self.image_path



class Renderer: 

    # Questa classe deve gestire entrambe le tipologie di animazione
    
    def __init__(self):
        # TODO: l'oggetto che serve per poter scrivere sul video dovrebbe essere passato qui 
        # ...
        self.animations = dict()

    def add_animation(self, animation_name:str, animation:Animation): 
        assert animation_name not in self.animations, "Animation names must be unique."
        self.animations[animation_name] = animation

    def start_animation(self, animation_name:str): self.animations[animation_name].start()

    def end_animation(self, animation_name:str): self.animations[animation_name].end()

    def is_active(self, animation_name:str) -> bool: return self.animations[animation_name].is_active()








# ==================== TEST ====================

if __name__ == "__main__": 
    fireball = TimedAnimation(1, ANIMATION_PATH / "flame_transparent.png")
    chidori = TimedAnimation(3, ANIMATION_PATH / "TODO")

    # No human being is able to do weird hand signs so fast but I allow to cast multiple jutsu 
    start = time.time()
    fireball.start()
    time.sleep(1)
    chidori.start()
    print("Both animation started")
    while fireball.is_active(): pass
    fireball.end()
    print("fireball finished after", time.time() - start)
    while chidori.is_active(): pass
    chidori.end()
    print("chidori finished, ", time.time() - start)

    # Test as a sequence (and test restart of the same object)
    start = time.time()
    fireball.start()
    while fireball.is_active(): pass
    print("fireball finished after", time.time() - start)

    start = time.time()
    chidori.start()
    while chidori.is_active(): pass
    print("chidori finished, ", time.time() - start)

    # Test handling via Renderer
    renderer = Renderer()
    renderer.add_animation("fireball", fireball)
    renderer.add_animation("chidori", chidori)
    renderer.start_animation("fireball")
    while renderer.is_active("fireball"): pass
    renderer.end_animation("fireball")


