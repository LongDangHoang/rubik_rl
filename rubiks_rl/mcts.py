import time
import numpy as np
import torch
import torch.nn as nn
import rubiks_rl.constants as constants

from rubiks_rl.rubik54 import Rubik54
from rubiks_rl.models import RLRubikModel, LMRubikModel
from rubiks_rl.world import get_depth_1_lookup_of_state

from timeit import default_timer as timer
from dataclasses import dataclass
from typing import List, Optional, Callable

@dataclass
class Node:
    count_times_explored: np.ndarray
    maximal_value: np.ndarray
    virtual_loss: np.ndarray
    prior_probability: np.ndarray
    prior_state_value: float
    cube_state: np.ndarray

    is_leaf: bool = True
    children: Optional[List["Node"]] = None

    
class MCTSRubik54:

    CUBE = Rubik54()

    def __init__(
        self,
        cube_state: np.ndarray,
        exploration_coef: float=0.01,
        virtual_loss_coef: float=0.01,
        verbose: bool=False,
    ) -> None:
        self.exploration_coef = exploration_coef
        self.virtual_loss_coef = virtual_loss_coef
        self.verbose = verbose
        
        v, p = self.state_value_next_action_prob_function(
            np.expand_dims(cube_state, 0), 
            np.array([]).reshape((1, 0))
        )
        self.root = Node(
            cube_state=cube_state,
            prior_probability=p[0],
            prior_state_value=v[0],
            count_times_explored=np.zeros(12),
            maximal_value=np.zeros(12),
            virtual_loss=0,
        )

    def simulate_once(self):
        node = self.root
        visited_nodes: List[Node] = []
        chosen_actions: List[int] = []
        
        while not node.is_leaf:
            if self.verbose:
                print(f"========= At node ID {id(node)} ==========")

            exploration_term: np.ndarray = (
                self.exploration_coef
                * node.prior_probability 
                * np.sqrt(node.count_times_explored.sum())
                * (1 + node.count_times_explored)
            )

            exploitation_term: np.ndarray = (
                node.maximal_value - node.virtual_loss
            )

            chosen_action = np.argmax(exploration_term + exploitation_term)
            chosen_actions.append(chosen_action)
            visited_nodes.append(node)

            node.virtual_loss[chosen_action] += self.virtual_loss_coef
            node = node.children[chosen_action]

            if self.verbose:
                print("Exploration terms:", exploration_term)
                print("Exploitation terms:", exploitation_term)
                print("Chosen action:", chosen_action)
                print("")

        else:
            node.is_leaf = False
            states = get_depth_1_lookup_of_state(
                np.expand_dims(node.cube_state, 0)
            )
            actions = np.concatenate([
                np.tile(np.array(chosen_actions), (12, 1)),
                np.expand_dims(np.arange(12), 1)
            ], axis=1)

            vs, ps = self.state_value_next_action_prob_function(states, actions)
            for action_idx in range(12):
                node.children[action_idx] = Node(
                    cube_state=states[action_idx],
                    prior_probability=ps[action_idx],
                    prior_state_value=vs[action_idx],
                    count_times_explored=np.zeros(12),
                    maximal_value=np.zeros(12),
                    virtual_loss=0,
                )
        
            expanded_node = node

            if self.verbose:
                print(f"========= Expanding node ID {id(node)} ==========")
                print("Prior values for children:", vs)
                print("Prior probabilities for children:\n", ps)
                print("")

        for node, action_idx in zip(visited_nodes, chosen_actions):
            node.count_times_explored[action_idx] += 1
            node.maximal_value[action_idx] = np.max(expanded_node.prior_state_value, node.maximal_value[action_idx])
            node.virtual_loss[action_idx] -= self.virtual_loss_coef

        return expanded_node, chosen_actions

    def search(self, budget_time: int):
        last_leaf_node = None
        last_path = None
        start = timer()
        while (timer() - start < budget_time):
            if (last_leaf_node is not None) and np.all(
                last_leaf_node.cube_state == self.CUBE.get_solved_state()
            ):
                return last_path
            
            last_leaf_node, last_path = self.simulate_once()

        return self.root.maximal_value.argmax()

    def get_tree_after_move(self, move_idx: int):
        self.root = self.root.children[move_idx]

    def state_value_next_action_prob_function(self, states: np.ndarray, actions: np.ndarray):
        raise NotImplementedError


class MCTSRL(MCTSRubik54):
    def __init__(self, model: RLRubikModel, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def state_value_next_action_prob_function(self, states: np.ndarray, chosen_actions: np.ndarray):
        with torch.no_grad():
            model_dtype = next(iter(self.model.parameters())).dtype
            states = torch.as_tensor(states, dtype=model_dtype, device=self.model.device)
            v, p = self.model(states)
            return v.cpu().numpy(), p.cpu().numpy()


class MCTSLM(MCTSRubik54):
    def __init__(self, model: LMRubikModel, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def state_value_next_action_prob_function(self, states: np.ndarray, chosen_actions: np.ndarray):
        assert chosen_actions.shape[0] == states.shape[0]
        with torch.no_grad():
            tokens = np.concatenate([
                (
                    self.root.cube_state
                    if chosen_actions.shape[1] > 0
                    else states
                ).argmax(axis=1) + 1,
                chosen_actions + 7,
                np.ones((
                    chosen_actions.shape[0], 
                    constants.MAX_SEQUENCE_LENGTH
                    - chosen_actions.shape[1]       # actions so far
                    - 54                            # cube states
                )) * constants.PADDING_IDX
            ], axis=1)
            assert len(tokens.shape) == 2 and tokens.shape[1] == constants.MAX_SEQUENCE_LENGTH and tokens.shape[0] == chosen_actions.shape[0]

            tokens = torch.as_tensor(tokens, dtype=torch.long).to(self.model.device)
            mask = torch.triu(torch.ones(constants.MAX_SEQUENCE_LENGTH, constants.MAX_SEQUENCE_LENGTH))
            mask[:54, :54] = 0
            mask = mask.to(self.model.device)

            p = self.model(states)[:, 53 + chosen_actions.shape[1], :]
            vs = p[:, 12]
            ps = p[:, :12] / p.sum(axis=1)
            return vs.cpu().numpy(), ps.cpu().numpy()
