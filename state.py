from enum import Enum

import numpy as np

class Color:
    RED = 1
    BLUE = 2
    WHITE = 3
    ORANGE = 4
    YELLOW = 5
    GREEN = 6

    @classmethod
    def one_hot(cls, idx: int) -> np.ndarray:
        v = np.zeros(6)
        v[idx - 1] = 1
        return v

class Rubik54:
    """
    Manages the state representing the cube and operations on the state

    State is represented as a 54 x 6 matrix, with a one-hot color vector 
    per sticker on the cube

    We keep the western convention in rubik:
    - front: green
    - left: orange
    - right: red
    - top: white
    - down: yellow
    - back: blue
    """
    # define the solved state as a 54 by 6 matrix
    # we go in the order front -> right -> back -> left -> top -> down
    SOLVED_STATE = np.array([
        # front 
        *([Color.one_hot(Color.GREEN)] * 9),
        # right
        *([Color.one_hot(Color.RED)] * 9),
        # back
        *([Color.one_hot(Color.BLUE)] * 9),
        # left
        *([Color.one_hot(Color.ORANGE)] * 9),
        # top
        *([Color.one_hot(Color.WHITE)] * 9),
        # down
        *([Color.one_hot(Color.YELLOW)] * 9),
    ])
    
    # for each face, index starts 1 in the lower left, then increasing in the direction of going right, then up
    # define offset for each face (i.e. 3rd index in the back is 21st in total)
    FRONT_OFFSET = 0
    RIGHT_OFFSET = 9
    BACK_OFFSET = 18
    LEFT_OFFSET = 27
    TOP_OFFSET = 36
    DOWN_OFFSET = 45

    # mapping for each turn. This is a very large dictionary that we will convert
    # to a numpy array
    F = {
        LEFT_OFFSET + 2: TOP_OFFSET,
        LEFT_OFFSET + 5: TOP_OFFSET + 1,
        LEFT_OFFSET + 8: TOP_OFFSET + 2,
        TOP_OFFSET: RIGHT_OFFSET + 6,
        TOP_OFFSET + 1: RIGHT_OFFSET + 3,
        TOP_OFFSET + 2: RIGHT_OFFSET,
        RIGHT_OFFSET + 6: DOWN_OFFSET + 8,
        RIGHT_OFFSET + 3: DOWN_OFFSET + 7,
        RIGHT_OFFSET: DOWN_OFFSET + 6,
        DOWN_OFFSET + 8: LEFT_OFFSET + 2,
        DOWN_OFFSET + 7: LEFT_OFFSET + 5,
        DOWN_OFFSET + 6: LEFT_OFFSET + 8,
        FRONT_OFFSET: FRONT_OFFSET + 6,
        FRONT_OFFSET + 6: FRONT_OFFSET + 8,
        FRONT_OFFSET + 8: FRONT_OFFSET + 2,
        FRONT_OFFSET + 2: FRONT_OFFSET,
        FRONT_OFFSET + 1: FRONT_OFFSET + 3,
        FRONT_OFFSET + 3: FRONT_OFFSET + 7,
        FRONT_OFFSET + 7: FRONT_OFFSET + 5,
        FRONT_OFFSET + 5: FRONT_OFFSET + 1
    }
    F_PRIME = {
        v: k for k, v in F.items()
    }

    R = {
        TOP_OFFSET + 2: BACK_OFFSET + 6,
        TOP_OFFSET + 5: BACK_OFFSET + 3,
        TOP_OFFSET + 8: BACK_OFFSET,
        BACK_OFFSET + 6: DOWN_OFFSET + 8,
        BACK_OFFSET + 3: DOWN_OFFSET + 5,
        BACK_OFFSET: DOWN_OFFSET + 2,
        DOWN_OFFSET + 8: FRONT_OFFSET + 2,
        DOWN_OFFSET + 5: FRONT_OFFSET + 5,
        DOWN_OFFSET + 2: FRONT_OFFSET + 8,
        FRONT_OFFSET + 2: TOP_OFFSET + 2,
        FRONT_OFFSET + 5: TOP_OFFSET + 5,
        FRONT_OFFSET + 8: TOP_OFFSET + 8,
        RIGHT_OFFSET: RIGHT_OFFSET + 6,
        RIGHT_OFFSET + 6: RIGHT_OFFSET + 8,
        RIGHT_OFFSET + 8: RIGHT_OFFSET + 2,
        RIGHT_OFFSET + 2: RIGHT_OFFSET,
        RIGHT_OFFSET + 1: RIGHT_OFFSET + 3,
        RIGHT_OFFSET + 3: RIGHT_OFFSET + 7,
        RIGHT_OFFSET + 7: RIGHT_OFFSET + 5,
        RIGHT_OFFSET + 5: RIGHT_OFFSET + 1
    }
    R_PRIME = {
        v: k for k, v in R.items()
    }

    B = {
        TOP_OFFSET + 6: LEFT_OFFSET,
        TOP_OFFSET + 7: LEFT_OFFSET + 3,
        TOP_OFFSET + 8: LEFT_OFFSET + 6,
        LEFT_OFFSET: DOWN_OFFSET + 2,
        LEFT_OFFSET + 3: DOWN_OFFSET + 1,
        LEFT_OFFSET + 6: DOWN_OFFSET,
        DOWN_OFFSET + 2: RIGHT_OFFSET + 8,
        DOWN_OFFSET + 1: RIGHT_OFFSET + 5,
        DOWN_OFFSET: RIGHT_OFFSET + 2,
        RIGHT_OFFSET + 8: TOP_OFFSET + 6,
        RIGHT_OFFSET + 5: TOP_OFFSET + 7,
        RIGHT_OFFSET + 2: TOP_OFFSET + 8,
        BACK_OFFSET: BACK_OFFSET + 6,
        BACK_OFFSET + 6: BACK_OFFSET + 8,
        BACK_OFFSET + 8: BACK_OFFSET + 2,
        BACK_OFFSET + 2: BACK_OFFSET,
        BACK_OFFSET + 1: BACK_OFFSET + 3,
        BACK_OFFSET + 3: BACK_OFFSET + 7,
        BACK_OFFSET + 7: BACK_OFFSET + 5,
        BACK_OFFSET + 5: BACK_OFFSET + 1
    }
    B_PRIME = {
        v: k for k, v in B.items()
    }

    L = {
        BACK_OFFSET + 2: TOP_OFFSET + 6,
        BACK_OFFSET + 5: TOP_OFFSET + 3,
        BACK_OFFSET + 8: TOP_OFFSET,
        TOP_OFFSET + 6: FRONT_OFFSET + 6,
        TOP_OFFSET + 3: FRONT_OFFSET + 3,
        TOP_OFFSET: FRONT_OFFSET,
        FRONT_OFFSET + 6: DOWN_OFFSET + 6,
        FRONT_OFFSET + 3: DOWN_OFFSET + 3,
        FRONT_OFFSET: DOWN_OFFSET,
        DOWN_OFFSET + 6: BACK_OFFSET + 2,
        DOWN_OFFSET + 3: BACK_OFFSET + 5,
        DOWN_OFFSET: BACK_OFFSET + 8,
        LEFT_OFFSET: LEFT_OFFSET + 6,
        LEFT_OFFSET + 6: LEFT_OFFSET + 8,
        LEFT_OFFSET + 8: LEFT_OFFSET + 2,
        LEFT_OFFSET + 2: LEFT_OFFSET,
        LEFT_OFFSET + 1: LEFT_OFFSET + 3,
        LEFT_OFFSET + 3: LEFT_OFFSET + 7,
        LEFT_OFFSET + 7: LEFT_OFFSET + 5,
        LEFT_OFFSET + 5: LEFT_OFFSET + 1
    }
    L_PRIME = {
        v: k for k, v in L.items()
    }

    T = {
        LEFT_OFFSET + 6: BACK_OFFSET + 6,
        LEFT_OFFSET + 7: BACK_OFFSET + 7,
        LEFT_OFFSET + 8: BACK_OFFSET + 8,
        BACK_OFFSET + 6: RIGHT_OFFSET + 6,
        BACK_OFFSET + 7: RIGHT_OFFSET + 7,
        BACK_OFFSET + 8: RIGHT_OFFSET + 8,
        RIGHT_OFFSET + 6: FRONT_OFFSET + 6,
        RIGHT_OFFSET + 7: FRONT_OFFSET + 7,
        RIGHT_OFFSET + 8: FRONT_OFFSET + 8,
        FRONT_OFFSET + 6: LEFT_OFFSET + 6,
        FRONT_OFFSET + 7: LEFT_OFFSET + 7,
        FRONT_OFFSET + 8: LEFT_OFFSET + 8,
        TOP_OFFSET: TOP_OFFSET + 6,
        TOP_OFFSET + 6: TOP_OFFSET + 8,
        TOP_OFFSET + 8: TOP_OFFSET + 2,
        TOP_OFFSET + 2: TOP_OFFSET,
        TOP_OFFSET + 1: TOP_OFFSET + 3,
        TOP_OFFSET + 3: TOP_OFFSET + 7,
        TOP_OFFSET + 7: TOP_OFFSET + 5,
        TOP_OFFSET + 5: TOP_OFFSET + 1
    }
    T_PRIME = {
        v: k for k, v in T.items()
    }

    D = {
        LEFT_OFFSET: FRONT_OFFSET,
        LEFT_OFFSET + 1: FRONT_OFFSET + 1,
        LEFT_OFFSET + 2: FRONT_OFFSET + 2,
        FRONT_OFFSET: RIGHT_OFFSET,
        FRONT_OFFSET + 1: RIGHT_OFFSET + 1,
        FRONT_OFFSET + 2: RIGHT_OFFSET + 2,
        RIGHT_OFFSET: BACK_OFFSET,
        RIGHT_OFFSET + 1: BACK_OFFSET + 1,
        RIGHT_OFFSET + 2: BACK_OFFSET + 2,
        BACK_OFFSET: LEFT_OFFSET,
        BACK_OFFSET + 1: LEFT_OFFSET + 1,
        BACK_OFFSET + 2: LEFT_OFFSET + 2,
        DOWN_OFFSET: DOWN_OFFSET + 6,
        DOWN_OFFSET + 6: DOWN_OFFSET + 8,
        DOWN_OFFSET + 8: DOWN_OFFSET + 2,
        DOWN_OFFSET + 2: DOWN_OFFSET,
        DOWN_OFFSET + 1: DOWN_OFFSET + 3,
        DOWN_OFFSET + 3: DOWN_OFFSET + 7,
        DOWN_OFFSET + 7: DOWN_OFFSET + 5,
        DOWN_OFFSET + 5: DOWN_OFFSET + 1
    }
    D_PRIME = {
        v: k for k, v in D.items()
    }

    def __init__(self):
        self.state = self.get_solved_state()
        self.turns = {
            turn_name: self.get_turn_state_idx(turn_swap_dict)
            for turn_name, turn_swap_dict in [
                ("F", self.F),
                ("R", self.R),
                ("B", self.B),
                ("L", self.L),
                ("T", self.T),
                ("D", self.D),
                ("F_PRIME", self.F_PRIME),
                ("R_PRIME", self.R_PRIME),
                ("B_PRIME", self.B_PRIME),
                ("L_PRIME", self.L_PRIME),
                ("T_PRIME", self.T_PRIME),
                ("D_PRIME", self.D_PRIME),
            ]
        }
    
    def get_solved_state(self):
        """Return the state which we consider the cube is solve"""
        return self.SOLVED_STATE

    def get_turn_state_idx(self, swap_dict):
        """Return an index into the state as a result of a turn from index swap dictionary"""
        idx_arr = np.arange(54)
        for k, v in swap_dict:
            idx_arr[k], idx_arr[v] = idx_arr[v], idx_arr[k]
        return idx_arr

    def change_state_after_turn(self, state: np.ndarray, turn_name: str):
        return state[self.turns[turn_name]]

    def is_solved_state(self, state: np.ndarray):
        return np.all(state == self.SOLVED_STATE)

    def visualise_state(self):
        pass
