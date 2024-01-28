import unittest
import numpy as np
import rubiks_rl.constants as constants

from rubiks_rl.colors import Color
from rubiks_rl.datasets import LMDataset
from rubiks_rl.rubik54 import Rubik54

class LMDatasetTest(unittest.TestCase):
    CUBE = Rubik54()

    @classmethod
    def setUpClass(cls):
        cls.dataset = LMDataset(num_observations=10)
        cls.dataset.set_data(seed=12321)

    def test_sequence_length(self):
        for idx in range(len(self.dataset)):
            tokens, seq_length_used = self.dataset[idx]
            count_0 = (tokens == 0).sum()
            lhs = count_0 + self.dataset.data["use_seq_length"][idx] + 54 + 1 # 54 cube states, 1 stop sequence
            self.assertEqual(lhs, constants.MAX_SEQUENCE_LENGTH)
            self.assertEqual(len(tokens), lhs)
            self.assertEqual(seq_length_used, self.dataset.data["use_seq_length"][idx])
            self.assertGreater(count_0, 0)

    def test_data_state(self):
        for idx in range(len(self.dataset)):
            tokens, _ = self.dataset[idx]
            cube_states = [int(s - 1) for s in tokens[:54]]
            cube_states = np.array([Color.one_hot(s) for s in cube_states])
            moves = [int(mv - 7) for mv in tokens[54:] if mv not in (constants.PADDING_IDX, constants.STOP_SEQ_IDX)]
            
            for mv in moves:
                cube_states = cube_states[self.dataset.cube.turn_mat[mv]]
            
            assert np.all(cube_states == self.dataset.cube.get_solved_state())
    