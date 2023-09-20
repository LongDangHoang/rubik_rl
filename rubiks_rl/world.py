"""
Utility method to generate batch of training/evaluation samples
"""
import torch
import numpy as np
import torch.nn as nn

from rubiks_rl.rubik54 import Rubik54

from typing import Callable, Tuple

CUBE = Rubik54()

def get_n_cubes_k_scrambles(
    num_cubes: int, 
    max_depth_scramble: int,
) -> np.ndarray:
    """
    Return a (max_depth_scramble*num_cubes, 54, 6) tensor 
    reprensenting num_cubes states over max_depth_scramble scrambles

    The k-th cube's state is at position k-1, max_depth_scramble + k-1, 2 * max_depth_sramble + k-1
    """
    solved_state = CUBE.get_solved_state()
    data = np.array([solved_state] * num_cubes)
    data_lst = []

    for _ in range(max_depth_scramble):
        move = np.random.randint(12, size=(num_cubes))
        turn_mat = CUBE.turn_mat[move]
        data_lst.append(
            np.take_along_axis(data, np.expand_dims(turn_mat, 2), axis=1)
        )
        data = data_lst[-1]

    data = np.concatenate(data_lst)
    return data


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
    num_states = cube_state.size(0)
    cube_state = np.repeat(cube_state, 12, axis=0)
    move = np.tile(np.arange(12), num_states)
    turn_mat = CUBE.turn_mat[move]

    # make the move
    cube_state = np.take_along_axis(cube_state, np.expand_dims(turn_mat, 2), axis=1)

    return cube_state


def find_best_move_and_value_from(
    cube_state: np.ndarray,
    evaluate_fn: Callable,
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

    # add in the reward
    cube_state_values = cube_state_values - 1 + 2 * cube_state_values == CUBE.SOLVED_STATE

    # take maximum and action
    best_action = cube_state_values.reshape((-1, 12)).argmax(axis=1)
    best_cube_state_values = cube_state_values[best_action]
    
    return best_action, best_cube_state_values


def model_evaluate(cube_state: np.ndarray, model: nn.Module, device: torch.DeviceObjType=None, batch_size: int=32) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    values = []
    with torch.no_grad():
        for batch in cube_state[::batch_size]:
            batch = torch.Tensor(batch, device=device)
            values.extend(model(batch)[:, 0])
    
    return np.array(values)
