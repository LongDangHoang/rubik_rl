import numpy as np
import rubiks_rl.constants as constants

from torch.utils.data import Dataset
from rubiks_rl.rubik54 import Rubik54
from rubiks_rl.world import get_n_cubes_k_scrambles
from typing import Optional


class LMDataset(Dataset):
    def __init__(self, num_observations: int=10_000):
        super().__init__()
        self.num_observations = num_observations
        self.cube = Rubik54()
        self.max_sequence_length = constants.MAX_SEQUENCE_LENGTH - len(self.cube.get_solved_state()) - 1
        assert self.max_sequence_length > 50, "Max sequence length is less than 50"
        self.rng = np.random.default_rng(seed=314)

        self.set_data(seed=self.rng.integers(0, 1000))

    def set_data(self, seed: Optional[int]=None):
        if seed is None:
            seed = self.rng.integers(0, 100_000)

        self.data = get_n_cubes_k_scrambles(
            num_cubes=self.num_observations,
            max_depth_scramble=self.max_sequence_length,
            seed=seed
        )

        seq_length_rng = np.random.default_rng(seed)
        self.data["use_seq_length"] = seq_length_rng.integers(
            0, self.max_sequence_length, size=(self.num_observations)
        )

    def __len__(self):
        return self.num_observations
    
    def __getitem__(self, index: int):
        seq_length_used = self.data["use_seq_length"][index]
        scrambled_cube = self.data["data"][
            index + self.num_observations * seq_length_used
        ].idxmax(axis=1)
        move_list = self.data["moves"][index][:seq_length_used+1]

        # convert index into face color (0-5) and index into move (0 - 11) 
        # into index into embedding (size 20)
        # and reverse the string so we start by unscrambling
        # also add padding up to seq length max
        tokens = np.concatenate(scrambled_cube + 1, np.fliplr(move_list + 7))
        assert not np.any(tokens == constants.PADDING_IDX)
        pads = np.ones(self.max_sequence_length - len(tokens)) * constants.PADDING_IDX
        tokens = np.concatenate(tokens, pads)
        assert len(tokens) == self.max_sequence_length

        return tokens


class RLDataset:
    """Refer to code run using the RL notebook. Re-implementing it here is not necessary"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
