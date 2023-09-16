import numpy as np
import pythreejs as p3

from rubiks_rl.colors import Color

from typing import Dict

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
    # to a numpy array. The dictionary goes from source->target (after move, target 
    # will use the color of source)
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
        BACK_OFFSET + 6: DOWN_OFFSET + 2,
        BACK_OFFSET + 3: DOWN_OFFSET + 5,
        BACK_OFFSET: DOWN_OFFSET + 8,
        DOWN_OFFSET + 2: FRONT_OFFSET + 2,
        DOWN_OFFSET + 5: FRONT_OFFSET + 5,
        DOWN_OFFSET + 8: FRONT_OFFSET + 8,
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

    # Define cubelets
    FRONT_TOP_LEFT_CORNER = {
        "FRONT": FRONT_OFFSET + 6,
        "TOP": TOP_OFFSET,
        "LEFT": LEFT_OFFSET + 8,
    }

    FRONT_TOP_RIGHT_CORNER = {
        "FRONT": FRONT_OFFSET + 8,
        "TOP": TOP_OFFSET + 2,
        "RIGHT": RIGHT_OFFSET + 6,
    }

    FRONT_DOWN_LEFT_CORNER = {
        "FRONT": FRONT_OFFSET,
        "DOWN": DOWN_OFFSET + 6,
        "LEFT": LEFT_OFFSET + 2,
    }

    FRONT_DOWN_RIGHT_CORNER = {
        "FRONT": FRONT_OFFSET + 2,
        "DOWN": DOWN_OFFSET + 8,
        "RIGHT": RIGHT_OFFSET,
    }

    BACK_TOP_LEFT_CORNER = {
        "BACK": BACK_OFFSET + 8,
        "TOP": TOP_OFFSET + 6,
        "LEFT": LEFT_OFFSET + 6,
    }

    BACK_TOP_RIGHT_CORNER = {
        "BACK": BACK_OFFSET + 6,
        "TOP": TOP_OFFSET + 8,
        "RIGHT": RIGHT_OFFSET + 8,
    }

    BACK_DOWN_LEFT_CORNER = {
        "BACK": BACK_OFFSET + 2,
        "DOWN": DOWN_OFFSET,
        "LEFT": LEFT_OFFSET,
    }

    BACK_DOWN_RIGHT_CORNER = {
        "BACK": BACK_OFFSET,
        "DOWN": DOWN_OFFSET + 2,
        "RIGHT": RIGHT_OFFSET + 2,
    }

    FRONT_DOWN_EDGE = {
        "FRONT": FRONT_OFFSET + 1,
        "DOWN": DOWN_OFFSET + 7,
    }

    FRONT_LEFT_EDGE = {
        "FRONT": FRONT_OFFSET + 3,
        "LEFT": LEFT_OFFSET + 5,
    }

    FRONT_RIGHT_EDGE = {
        "FRONT": FRONT_OFFSET + 5,
        "RIGHT": RIGHT_OFFSET + 3,
    }

    FRONT_TOP_EDGE = {
        "FRONT": FRONT_OFFSET + 7,
        "TOP": TOP_OFFSET + 1,
    }

    LEFT_TOP_EDGE = {
        "TOP": TOP_OFFSET + 3,
        "LEFT": LEFT_OFFSET + 7,
    }

    LEFT_DOWN_EDGE = {
        "DOWN": DOWN_OFFSET + 3,
        "LEFT": LEFT_OFFSET + 1,
    }

    RIGHT_TOP_EDGE = {
        "TOP": TOP_OFFSET + 5,
        "RIGHT": RIGHT_OFFSET + 7,
    }

    RIGHT_DOWN_EDGE = {
        "DOWN": DOWN_OFFSET + 5,
        "RIGHT": RIGHT_OFFSET + 1,
    }

    BACK_DOWN_EDGE = {
        "BACK": BACK_OFFSET + 1,
        "DOWN": DOWN_OFFSET + 1,
    }

    BACK_LEFT_EDGE = {
        "BACK": BACK_OFFSET + 5,
        "LEFT": LEFT_OFFSET + 3,
    }

    BACK_RIGHT_EDGE = {
        "BACK": BACK_OFFSET + 3,
        "RIGHT": RIGHT_OFFSET + 5,
        "LEFT": None,
    }

    BACK_TOP_EDGE = {
        "BACK": BACK_OFFSET + 7,
        "TOP": TOP_OFFSET + 7,
    }

    MID_FRONT = {"FRONT": FRONT_OFFSET + 4}
    MID_RIGHT = {"RIGHT": RIGHT_OFFSET + 4}
    MID_TOP = {"TOP": TOP_OFFSET + 4}
    MID_DOWN = {"DOWN": DOWN_OFFSET + 4}
    MID_BACK = {"BACK": BACK_OFFSET + 4}
    MID_LEFT = {"LEFT": LEFT_OFFSET + 4}
    CENTER = {}

    # i point to right, j point to top, k point to front
    # and 0, 0, 0 is back-down-left corner
    POSITION_IJK_TO_CUBELET = {
        (0, 0, 0): BACK_DOWN_LEFT_CORNER,
        (0, 0, 1): LEFT_DOWN_EDGE,
        (0, 0, 2): FRONT_DOWN_LEFT_CORNER,
        (0, 1, 0): BACK_LEFT_EDGE,
        (0, 1, 1): MID_LEFT,
        (0, 1, 2): FRONT_LEFT_EDGE,
        (0, 2, 0): BACK_TOP_LEFT_CORNER,
        (0, 2, 1): LEFT_TOP_EDGE,
        (0, 2, 2): FRONT_TOP_LEFT_CORNER,
        (1, 0, 0): BACK_DOWN_EDGE,
        (1, 0, 1): MID_DOWN,
        (1, 0, 2): FRONT_DOWN_EDGE,
        (1, 1, 0): MID_BACK,
        (1, 1, 1): CENTER,
        (1, 1, 2): MID_FRONT,
        (1, 2, 0): BACK_TOP_EDGE,
        (1, 2, 1): MID_TOP,
        (1, 2, 2): FRONT_TOP_EDGE,
        (2, 0, 0): BACK_DOWN_RIGHT_CORNER,
        (2, 0, 1): RIGHT_DOWN_EDGE,
        (2, 0, 2): FRONT_DOWN_RIGHT_CORNER,
        (2, 1, 0): BACK_RIGHT_EDGE,
        (2, 1, 1): MID_RIGHT,
        (2, 1, 2): FRONT_RIGHT_EDGE,
        (2, 2, 0): BACK_TOP_RIGHT_CORNER,
        (2, 2, 1): RIGHT_TOP_EDGE,
        (2, 2, 2): FRONT_TOP_RIGHT_CORNER,
    }

    TURN_IDX_TO_STR_SWAP_DICT = {
        1: ("F", F),
        2: ("R", R),
        3: ("B", B),
        4: ("L", L),
        5: ("T", T),
        6: ("D", D),
        7: ("F_PRIME", F_PRIME),
        8: ("R_PRIME", R_PRIME),
        9: ("B_PRIME", B_PRIME),
        10: ("L_PRIME", L_PRIME),
        11: ("T_PRIME", T_PRIME),
        12: ("D_PRIME", D_PRIME),
    }


    def __init__(self):
        self.state = self.get_solved_state()
        self.turns = {
            turn_name: self.get_turn_state_idx(turn_swap_dict)
            for turn_name, turn_swap_dict in self.TURN_IDX_TO_STR_SWAP_DICT.values()
        }
        self.turn_mat = np.array([self.turns[self.TURN_IDX_TO_STR_SWAP_DICT[i][0]] for i in range(12)])
    
    def get_solved_state(self):
        """Return the state which we consider the cube is solve"""
        return self.SOLVED_STATE

    def get_turn_state_idx(self, swap_dict: Dict[int, int]):
        """Return an index into the state as a result of a turn from index swap dictionary"""
        idx_arr = np.arange(54)
        for sticker_source, sticker_target in swap_dict.items():
            idx_arr[sticker_target] = sticker_source
        return idx_arr

    def change_state_after_turn(self, state: np.ndarray, turn_name: str):
        return state[self.turns[turn_name]]

    def is_solved_state(self, state: np.ndarray):
        return np.all(state == self.SOLVED_STATE)

    def visualise_state(self, state: np.ndarray) -> p3.Scene:
        # Create cube geometry
        geometry = p3.BoxGeometry(width=0.9, height=0.9, depth=0.9)

        # get colors for each cubelet

        FACES = ["FRONT", "RIGHT", "BACK", "LEFT", "TOP", "DOWN"]
        MESH_ORDERING = ["RIGHT", "LEFT", "TOP", "DOWN", "FRONT", "BACK"]

        cube_pieces = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    cubelet_face_state_lookup = self.POSITION_IJK_TO_CUBELET[(i, j, k)]
                    cubelet_colors = {
                        f: -1 if f not in cubelet_face_state_lookup else state[cubelet_face_state_lookup[f]].argmax()
                        for f in FACES
                    }
                    cubelet_str_colors = {
                        k: "purple" if color_idx == -1 else Color.get_str_name(color_idx).lower()
                        for k, color_idx in cubelet_colors.items()
                    }

                    materials = [p3.MeshBasicMaterial(color=cubelet_str_colors[face]) for face in MESH_ORDERING]

                    piece = p3.Mesh(geometry=geometry, material=materials)
                    piece.position = [i, j, k]
                    cube_pieces.append(piece)

        # Create a scene and add the cube pieces to it
        scene = p3.Scene(children=cube_pieces)
        return scene
