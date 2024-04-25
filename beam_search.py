import numpy as np
from CPMP.cpmp_ml import generate_data, Layout
from keras.models import Model
from copy import deepcopy

def init_session(lays: list[Layout]) -> list[tuple]:
    lays_size = len(lays)
    new_lays = []

    for i in range(lays_size):
        new_lays.append((lays[i], i))

    return new_lays

def add_solution(solutions: dict, lay_solution: list[tuple]) -> dict:
    solutions_size = len(solutions)

    if solutions_size == 0 or str(lay_solution[1]) in solutions:
        solutions.update({str(lay_solution[1]): [lay_solution[0]]})
    else:
        solutions[str(lay_solution[1])].append(lay_solution[0])
    
    return solutions

def prune_tree(index: list[int], lays: list[tuple]):
    new_lays = [lay for lay in lays if lays[1] not in index]

    return new_lays

def verify_solution(lays: list[tuple], solutions: dict) -> dict:
    lays_size = len(lays)
    index = []

    for i in range(lays_size):
        if lays[i][0].unsorted_stacks == 0:
            solutions = add_solution(solutions, lays[i])
            if i not in index: index.append(i)

    new_lays = prune_tree(index, lays)

    return new_lays, solutions

def get_model_lays(lays: list[tuple]) -> list:
    model_lays = [lay[0] for lay in lays]

    return np.stack(model_lays)

def expanded_lays(model: Model, lays: list[tuple]):
    lays_size = len(lays)
    child_lays = []

    model_lays = get_model_lays(lays)
    pred_lays = model.predict(model_lays, verbose= False)

    for k in range(lays_size):
        stack_size = len(lays[k][0].stacks)

        for i in range(stack_size):
            for j in range(stack_size):
                if i == j: continue

                new_lay = deepcopy(lays[k][0])
                new_lay.move((i, j))
                child_lays.append((new_lay, lays[k][1]))
    
    return child_lays, pred_lays



def beam_search(model: Model, lays: list[Layout] = None, H: int = 5, 
                w: int = 3 , threshold: float = 0.01) -> list:
    
    session_lays = init_session(lays)
    solutions = dict()
    
    while True:
        session_lays, solutions = verify_solution(session_lays, solutions)
        if len(session_lays) == 0: return solutions

        child_lays, pred_lays = expanded_lays(model, session_lays)
        