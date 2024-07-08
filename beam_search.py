import numpy as np
from keras.models import Model
from keras import backend as K
from CPMP.cpmp_ml import generate_random_layout, Layout
from CPMP_Model.Model_CPMP import load_model
from copy import deepcopy
import time

def get_ann_state(layout: Layout) -> np.ndarray:
    S=len(layout.stacks) # Cantidad de stacks

    #matriz de stacks
    b = 2. * np.ones([S,layout.H + 1]) # Matriz normalizada
    for i,j in enumerate(layout.stacks):
        b[i][layout.H-len(j) + 1:] = [k/layout.total_elements for k in j]
        b[i][0] = layout.is_sorted_stack(i)
    b.shape=(S,(layout.H + 1))
    
    return np.stack(b)

class BeemSearch():
    def __init__(self, model: Model, H: int, level_nodes: int, threshold: float) -> None:
        self.__model = model
        self.__H = H
        self.__level_nodes = level_nodes
        self.__threshold = threshold
        self.__session_lays_size = 0
        self.__init_states = None
        self.__solutions = dict()

    def __init_session_lays__(self, lays: np.ndarray[Layout]) -> dict:
        lays_size = lays.shape[0]
        self.__session_lays_size = lays_size
        session_lays = dict()

        for i in range(lays_size):
            session_lays.update({str(i): [lays[i]]})
        
        return session_lays

    def __add_solution__(self, state: Layout, case: str) -> None:
        if case not in self.__solutions:
            self.__solutions.update({case: [state]})
        elif case in self.__solutions:
            self.__solutions[case].append(state)

    def __prune_tree__(self, session_lays: dict, costs: np.ndarray, pred_lays: dict = None) -> dict:
        temp_session = deepcopy(session_lays)
        pred_lays = deepcopy(pred_lays)

        for case in session_lays:
            if costs[int(case)] != -1: 
                temp_session.pop(case)
                if pred_lays is not None and case in pred_lays: pred_lays.pop(case)

        return temp_session, pred_lays

    def __verify_solutions__(self, session_lays: dict, costs: np.ndarray, step: int, pred_lays: dict = None) -> np.ndarray:
        for case in session_lays:
            states = session_lays[case]
            states_size = len(states)

            for i in range(states_size):
                if states[i].unsorted_stacks == 0:
                    self.__add_solution__(states[i], case)
                    costs[int(case)] = step
        
        session_lays, pred_lays = self.__prune_tree__(session_lays, costs, pred_lays)

        if pred_lays is not None: return session_lays, pred_lays
        return session_lays
    
    def __get_model_lays__(self, session_lays: dict) -> np.ndarray:
        temp_session = deepcopy(session_lays)
        model_lays = []

        for case in temp_session:
            states = temp_session[case]
            states_size = len(states)

            for i in range(states_size):
                model_lays.append(get_ann_state(states[i]))
        
        return np.stack(model_lays)
    
    def __get_all_child_lays__(self, state: Layout, session_lays: dict, pred_lays: np.ndarray, case: str) -> tuple:
        S = len(state.stacks)
        new_pred_lay = []
        
        k = 0
        for i in range(S):
            for j in range(S):
                if i == j: continue
                temp = deepcopy(state)
                if temp.move((i, j)) is None: continue
                session_lays[case].append(temp)
                new_pred_lay.append(pred_lays[k])
                k += 1

        new_pred_lay = np.array(new_pred_lay)
        
        return session_lays, new_pred_lay

    def __expand_lays__(self, session_lays: dict) -> tuple:
        child_session_lays = dict()
        child_session_pred = dict()
        model_lays = self.__get_model_lays__(session_lays)

        pred_lays = self.__model.predict(model_lays, verbose= False)
        K.clear_session()
        
        k = 0
        for case in session_lays:
            child_session_lays.update({case: []})
            child_session_pred.update({case: []})

            states = session_lays[case]
            states_size = len(states)

            for i in range(states_size):
                child_session_lays, new_pred_lay = self.__get_all_child_lays__(states[i], child_session_lays, pred_lays[k], case)
                child_session_pred[case].append(new_pred_lay)
                k += 1

        return child_session_lays, child_session_pred
    
    def __sort_indices__(self, lst: list):
        lst = np.concatenate(lst)
        sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x], reverse= True)

        return sorted_indices[: self.__level_nodes]

    def __best_moves__(self, session_lays: dict, session_pred: dict, multiplication_pred: dict) -> dict:
        new_session_lays = dict()
        new_session_pred = dict()

        for case in session_lays:
            new_session_lays.update({case: []})
            new_session_pred.update({case: []})

            states, pred_lays = session_lays[case], np.concatenate(session_pred[case])
            states_size = len(states)
            temp_pred = self.__sort_indices__(multiplication_pred[case])

            new_pred = []
            for i in range(states_size):
                if pred_lays[i] >= self.__threshold and i in temp_pred:
                    new_session_lays[case].append(states[i])
                    new_pred.append(pred_lays[i])

            if len(new_pred) != 0: new_session_pred[case].append(np.array(new_pred))
            
        return new_session_lays, new_session_pred

    def __multiply_predictions__(self, session_pred_lays: dict, session_pred_child_lays: dict) -> np.ndarray:
        multiply_pred = dict()

        for case in session_pred_lays:
            multiply_pred.update({case: []})
            
            pred_lays = np.concatenate(session_pred_lays[case])
            child_pred_lays = session_pred_child_lays[case]

            for i in range(pred_lays.shape[0]):
                multiply_pred[case].append(child_pred_lays[i] * pred_lays[i])

        return multiply_pred
    
    def __count_unsorted_elements__(self, matrix: list) -> int:
        return sum(any(row[i] < row[i+1] for i in range(len(row)-1)) for row in matrix)

    def __find_lower_bound__(self, session_lays: dict) -> dict:
        lb = dict()

        for case in session_lays:
            unsorted_counts = [self.__count_unsorted_elements__(matrix.stacks) for matrix in session_lays[case]]
            lb.update({case: min(unsorted_counts)})

        return lb

    def __filter_lower_bound__(self, session_lays: dict, session_pred_lays: dict, lb: dict) -> tuple:
        new_session_lays = dict()
        new_session_pred = dict()

        for case in session_lays:
            new_session_lays.update({case: []})
            new_session_pred.update({case: []})

            states = session_lays[case]
            pred_lays = np.concatenate(session_pred_lays[case])
            states_size = len(states)

            new_pred = []
            for i in range(states_size):
                lay_lb = self.__count_unsorted_elements__(states[i].stacks)

                if lb[case] + self.__H >= lay_lb:
                    new_session_lays[case].append(states[i])
                    new_pred.append(pred_lays[i])
            
            new_session_pred[case].append(np.array(new_pred))
        
        return new_session_lays, new_session_pred

    def __verify_states_whitout_solution__(self, lays: np.ndarray[np.ndarray]) -> None:
        pass

    def __show_session_lays__(self, session_lays: dict, session_pred_lays: dict):
        for case in session_lays:
            states = session_lays[case]
            pred_lays = np.concatenate(session_pred_lays[case])
            states_size = len(states)

            print(f"{case =}")
            for i in range(states_size):
                print(f'   state: {states[i].stacks}')
                print(f'   prob: {pred_lays[i]}')


    def solve(self, lays: np.ndarray[Layout], max_steps: int) -> np.ndarray:
        start = time.time()
        steps = 0
        costs = np.full(lays.shape[0], -1)
        session_lays = self.__init_session_lays__(lays)
        
        session_lays = self.__verify_solutions__(session_lays, costs, steps)
        if len(session_lays) == 0: return costs

        session_lays, pred_lays = self.__expand_lays__(session_lays)
        session_lays, pred_lays = self.__best_moves__(session_lays, pred_lays, pred_lays)

        while True:
            steps += 1
            session_lays, pred_lays = self.__verify_solutions__(session_lays, costs, steps, pred_lays)
            if len(session_lays) == 0 or steps >= max_steps: break

            child_session_lays, pred_child_lays = self.__expand_lays__(session_lays)
            multiply_pred = self.__multiply_predictions__(pred_lays, pred_child_lays)
            session_lays, pred_lays = self.__best_moves__(child_session_lays, pred_child_lays, multiply_pred)

            lb = self.__find_lower_bound__(session_lays)
            session_lays, pred_lays = self.__filter_lower_bound__(session_lays, pred_lays, lb)
            print(f'Lower bound: {lb.values()}')

        end = time.time()

        print(f'\nExecution time: {round((end - start) / 60, 3)} minutes')

        return costs

    def get_init_states(self) -> np.ndarray:
        if self.__init_states is None: 
            print(f"No hay estados iniciales")
            return None
        
        return self.__init_states
    
    def get_solutions(self) -> list:
        if self.__solutions is None: 
            print(f"No hay resultados")
            return None
    
        return self.__solutions

    def set_model(self, new_model: Model, new_H) -> None:
        self.__model = new_model
        self.__model = new_H

    def set_level_nodes(self, new_level_nodes: int) -> None:
        self.__level_nodes = new_level_nodes
    
    def set_threshold(self, new_threshold: float) -> None:
        self.__threshold = new_threshold

def show_results(cases: list[Layout], solutions: dict):
    cases_size = len(cases)

    for i in range(cases_size):
        print(f'Case {i + 1}:')
        print(f'State: {cases[i].stacks}')

        case_solution = solutions[str(i)]
        case_solution_size = len(case_solution)

        for j in range(case_solution_size):
            print(f'    solution {j}:')
            print(f'        {case_solution[j].stacks}')


if __name__ == "__main__":
    model = load_model('./Models/model_5x5.keras')

    cases = [generate_random_layout(5, 5, 15) for _ in range(1000)]
    optimizer = BeemSearch(model, 5, 15, 0.0001)

    solutions = optimizer.solve(np.array(cases, dtype= object), 30)
    print(solutions)
    
        