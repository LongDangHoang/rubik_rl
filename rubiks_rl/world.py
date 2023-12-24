"""
Utility method to generate batch of training/evaluation samples
"""
import torch
import numpy as np
import torch.nn as nn

from rubiks_rl.rubik54 import Rubik54

from typing import Callable, Dict, List, Literal, Tuple

CUBE = Rubik54()

def get_n_cubes_k_scrambles(
    num_cubes: int, 
    max_depth_scramble: int,
    seed: int=314,
) -> Dict[Literal["data", "moves"], np.ndarray]:
    """
    Return a (max_depth_scramble*num_cubes, 54, 6) numpy array 
    reprensenting num_cubes states over max_depth_scramble scrambles, 
    and a (num_cubes, max_depth_scrambe) array of moves indices.

    The k-th cube's state is at position k-1, num_cubes + k-1, 2 * num_cubes + k-1 for 
    scramble depth 1, 2, 3, etc.
    """
    solved_state = CUBE.get_solved_state()
    data = np.array([solved_state] * num_cubes)
    data_lst = []
    moves_lst = []

    generator = np.random.default_rng(seed=seed)
    for _ in range(max_depth_scramble):
        move = generator.integers(12, size=(num_cubes))
        turn_mat = CUBE.turn_mat[move]
        data_lst.append(
            np.take_along_axis(data, np.expand_dims(turn_mat, 2), axis=1)
        )
        data = data_lst[-1]
        moves_lst.append(move)

    data = np.concatenate(data_lst)
    moves = np.stack(moves_lst, axis=1)
    return {"data": data, "moves": moves}


def get_weights_by_scrambling_distance(
    num_cubes: int,
    max_depth_scramble: int,
) -> np.ndarray:
    """
    Return a 1D array representing weights of each cube's state as the inverse of the 
    scrambling distance
    """
    return np.repeat(1 / np.arange(1, max_depth_scramble+1), num_cubes)


def get_depth_1_lookup_of_state(
    cube_state: np.ndarray
) -> np.ndarray:
    """
    Given a (num_states, 54, 6) tensor reprensenting num_states states,
    generate a (num_states * 12, 54, 6) tensor representing every possible move 
    from that state.

    The k-th state's subsequent states are at positions 12 * k + range(12)
    """
    # repeat state to pad out 12 times and likewise with move
    num_states = cube_state.shape[0]
    cube_state = np.repeat(cube_state, 12, axis=0)
    move = np.tile(np.arange(12), num_states)
    turn_mat = CUBE.turn_mat[move]

    # make the move
    cube_state = np.take_along_axis(cube_state, np.expand_dims(turn_mat, 2), axis=1)

    return cube_state


def find_best_move_and_value_from(
    cube_state: np.ndarray,
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the best move and associate state values from a set of states 
    by evaluating the values of the subsequent states using the modoel.

    Given a (num_states, 54, 6) tensor reprensenting num_states states,
    the state values would be of shape (num_states,) and the best actions 
    would be an index vector of shape (num_states,)
    """
    # generate look ahead by 1
    next_cube_state = get_depth_1_lookup_of_state(cube_state)

    # evaluate using a model
    cube_state_values = evaluate_fn(next_cube_state)
    assert len(cube_state_values.shape) == 1

    # add in the reward
    cube_state_values = cube_state_values - 1 + 2 * np.all(
        next_cube_state == np.expand_dims(CUBE.SOLVED_STATE, 0),
        axis=(1, 2)
    )

    # take maximum and action
    best_action = cube_state_values.reshape((-1, 12)).argmax(axis=1)
    best_cube_state_values = cube_state_values.reshape((-1, 12)).max(axis=1)
    
    return best_action, best_cube_state_values


def model_evaluate(cube_state: np.ndarray, model: nn.Module, device: torch.DeviceObjType=None, batch_size: int=32) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    values = []
    with torch.no_grad():
        for batch_idx in range(0, cube_state.shape[0], batch_size):
            batch = torch.as_tensor(
                cube_state[batch_idx:batch_idx+batch_size],
                device=device,
                dtype=torch.float32
            )
            v, _ = model.forward(batch)
            values.extend(v.cpu().numpy()[:, 0])
    
    return np.array(values)
