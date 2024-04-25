import numpy as np
from CPMP.cpmp_ml import generate_data, Layout

def init_session(lays: list[Layout]) -> list[tuple]:
    lays_size = len(lays)
    new_lays = []

    for i in range(lays_size):
        new_lays.append((lays[i], i))

    return new_lays

def verify_solution():

def beam_search(lays: list[Layout] = None, H: int = 5, 
                w: int = 3 , threshold: float = 0.01) -> list:
    
    session_lays = init_session(lays)
    solutions = []
    
    while True:
        