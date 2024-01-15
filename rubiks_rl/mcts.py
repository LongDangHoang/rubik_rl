import time
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass
from world import get_depth_1_lookup_of_state
from rubik54 import Rubik54
from typing import List, Optional

@dataclass
class Node:
    count_times_explored: np.ndarray
    maximal_value: np.ndarray
    virtual_loss: np.ndarray
    prior_probability: np.ndarray
    cube_state: np.ndarray

    is_leaf: bool = True
    children: Optional[List["Node"]] = None

    
class MCTSRubik54:

    EXPLORATION_COEF = 0.01
    VIRTUAL_LOSS_COEF = 0.01
    CUBE = Rubik54()

    def __init__(self, cube_state: np.ndarray, model: nn.Module) -> None:
        self.model = model
        self.device = model.device

        with torch.no_grad():
            model.eval()
            _, p = model(
                torch.as_tensor(
                    np.expand_dims(cube_state, 0),
                    dtype=torch.float32,
                    device=self.device
                )
            )
            p = p.cpu().numpy()

        self.root = Node(
            cube_state=cube_state,
            prior_probability=p,
            count_times_explored=np.zeros(12),
            maximal_value=np.ones(12)*-np.inf,
            virtual_loss=0,
        )


    def simulate_once(self):
        node = self.root
        visited_nodes: List[Node] = []
        chosen_actions: List[int] = []
        
        while not node.is_leaf:
            exploration_term: np.ndarray = (
                self.EXPLORATION_COEF
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

            node.virtual_loss[chosen_action] += self.VIRTUAL_LOSS_COEF
            node = node.children[chosen_action]

        else:
            node.is_leaf = False
            states = get_depth_1_lookup_of_state(
                np.expand_dims(node.cube_state, 0)
            )
            states = np.concatenate([
                states,
                np.expand_dims(node.cube_state, 0)
            ], axis=0)

            with torch.no_grad():
                self.model.eval()
                v, p = self.model(
                    torch.as_tensor(
                        states,
                        dtype=torch.float32,
                        device=self.device
                    )
                )
                p = p.cpu().numpy()
                v = v.cpu().numpy()

                for action_idx in range(12):
                    node.children[action_idx] = Node(
                        cube_state=states[action_idx],
                        prior_probability=p[action_idx],
                        count_times_explored=np.zeros(12),
                        maximal_value=np.ones(12)*-np.inf,
                        virtual_loss=0,
                    )
            
            expanded_node = node

        for node, action_idx in zip(visited_nodes, chosen_actions):
            node.count_times_explored[action_idx] += 1
            node.maximal_value[action_idx] = np.max(v[-1], node.maximal_value[action_idx])
            node.virtual_loss[action_idx] -= self.VIRTUAL_LOSS_COEF

        return expanded_node, chosen_actions

    def search(self, budget_time: int):
        start = time.time()
        if np.all(self.root.cube_state == self.CUBE.get_solved_state()):
            raise ValueError("Cube already solved, terminating search")

        last_leaf_node = None
        last_path = None

        while (time.time() - start < budget_time):
            if (last_leaf_node is not None) and np.all(
                last_leaf_node.cube_state == self.CUBE.get_solved_state()
            ):
                return last_path
            
            last_leaf_node, last_path = self.simulate_once()

        return self.root.maximal_value.argmax()


