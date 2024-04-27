import numpy as np
from CPMP.cpmp_ml import generate_random_layout, Layout
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from copy import deepcopy

def get_ann_state(layout: Layout) -> np.ndarray:
    S=len(layout.stacks) # Cantidad de stacks

    #matriz de stacks
    b = 2. * np.ones([S,layout.H + 1]) # Matriz normalizada
    for i,j in enumerate(layout.stacks):
        b[i][layout.H-len(j) + 1:] = [k/layout.total_elements for k in j]
        b[i][0] = layout.is_sorted_stack(i)
    b.shape=(S,(layout.H + 1))
    
    return np.stack(b)

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
    model_lays = [get_ann_state(lay[0]) for lay in lays]

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

def get_dimensions(lst: list) -> list:
    dimensions = []

    while isinstance(lst, list) or isinstance(lst, np.ndarray):  # Mientras el elemento sea una lista
        if isinstance(lst, list): dimensions.append(len(lst))
        elif isinstance(lst, np.ndarray):  # Agregar la longitud de la lista actual a las dimensiones
            lst = lst.tolist()
            dimensions.append(len(lst))

        if len(lst) == 0:
            # Si la lista está vacía, detener el bucle
            break
        lst = lst[0]  # Continuar con el primer elemento para revisar más niveles de anidamiento
    
    return dimensions 

def best_moves(lays: list[tuple], pred_lays: list[np.ndarray], w: int, threshold: int = 0.01) -> tuple:
    cases = len(pred_lays)
    new_lays = []
    new_pred = []

    cont = 0
    for k in range(cases):
        pred_size = pred_lays[k].shape[0]
        temp_w = 0

        for i in range(pred_size):
            if pred_lays[k][i] >= threshold and temp_w < w:
                new_lays.append(lays[cont])
                new_pred.append(pred_lays[k][i])
                temp_w += 1

            cont += 1

    return new_lays, new_pred

def multiply_predictions(pred_lays: list[np.ndarray], pred_child_lays: list[np.ndarray]) -> list[np.ndarray]:
    pred_lays_size = len(pred_lays)
    multiply_pred = []

    j = 0
    for k in range(pred_lays_size):
        stack_size = pred_lays[0].shape[0]
        
        for i in range(stack_size):
            multiply_pred.append(pred_child_lays[j] * pred_lays[k][i])
    
    return multiply_pred

def beam_search(model: Model, lays: list[Layout] = None, H: int = 5, 
                w: int = 3 , threshold: float = 0.01) -> list:
    
    session_lays = init_session(lays)
    solutions = dict()
    
    session_lays, solutions = verify_solution(session_lays, solutions)
    if len(session_lays) == 0: return solutions
    
    session_lays, pred_lays = expanded_lays(model, session_lays)

    while True:
        session_lays, solutions = verify_solution(session_lays, solutions)
        if len(session_lays) == 0: return solutions

        child_lays, pred_child_lays = expanded_lays(model, session_lays)
        multiply_pred = multiply_predictions(pred_lays, pred_child_lays)
        
        session_lays, pred_lays = best_moves(child_lays, multiply_pred, w, threshold)

    return solutions
    
input_model = Input(shape= (5, 6))
dense_1 = Dense(6)(input_model)
dense_2 = Dense(6)(dense_1)
flatten = Flatten()(dense_2)
dense_3 = Dense(20, activation= 'sigmoid')(flatten)

model = Model(inputs= input_model, outputs= dense_3)

beam_search(model, [generate_random_layout(5, 5, 15), generate_random_layout(5, 5, 15)])
        