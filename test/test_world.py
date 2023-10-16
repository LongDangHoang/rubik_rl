import unittest
import numpy as np

from rubiks_rl.colors import Color
from rubiks_rl.rubik54 import Rubik54
from rubiks_rl.world import (
    get_depth_1_lookup_of_state,
    get_n_cubes_k_scrambles,
    find_best_move_and_value_from
)

class WorldTest(unittest.TestCase):
    CUBE = Rubik54()
    TEST_SHORTHAND_STR = "L2 B2 F2 R' B' D F2 R2 F2 L2 D2 R D2 L' U' F L D' R' D' B' L' R2 B' D2 F2 L' R2 F' D L' R2 U' F' U L F' D' U2 R2 U2 B L' D B' F2 D F' L' R2" 

    def manual_check_short_hand(self):
        state = self.CUBE.apply_turn_shorthands(
            self.CUBE.get_solved_state(),
            self.TEST_SHORTHAND_STR
        )

        test_colors = [
            Color.BLUE, Color.GREEN, Color.WHITE,
            Color.GREEN, Color.GREEN, Color.WHITE,
            Color.BLUE, Color.GREEN, Color.ORANGE,
            Color.GREEN, Color.WHITE, Color.YELLOW,
            Color.BLUE, Color.RED, Color.BLUE,
            Color.YELLOW, Color.ORANGE, Color.ORANGE,
            Color.BLUE, Color.ORANGE, Color.RED,
            Color.RED, Color.BLUE, Color.BLUE,
            Color.WHITE, Color.ORANGE, Color.GREEN,
            Color.YELLOW, Color.RED, Color.RED,
            Color.YELLOW, Color.ORANGE, Color.YELLOW,
            Color.RED, Color.RED, Color.RED, 
            Color.WHITE, Color.ORANGE, Color.GREEN,
            Color.YELLOW, Color.WHITE, Color.BLUE,
            Color.WHITE, Color.WHITE, Color.BLUE,
            Color.GREEN, Color.YELLOW, Color.ORANGE,
            Color.WHITE, Color.YELLOW, Color.GREEN,
            Color.YELLOW, Color.RED, Color.ORANGE
        ]

        for test_color, values in zip(test_colors, state):
            color_idx = values.argmax()
            self.assertEqual(color_idx, test_color)

    def test_n_cubes_k_scrambles(self):
        test_n_cubes = 5
        test_d = 50

        data_moves_dict = get_n_cubes_k_scrambles(num_cubes=test_n_cubes, max_depth_scramble=test_d, seed=314) 
        for cube_idx in range(test_n_cubes):
            short_hand_str = ""
            for move_idx in range(test_d):
                move_str: str = self.CUBE.TURN_IDX_TO_STR_SWAP_DICT[
                    data_moves_dict["moves"][cube_idx][move_idx]
                ][0]
                move_str = move_str.replace("T", "U").replace("_PRIME", "'")
                short_hand_str += " " + move_str
                test_state = self.CUBE.apply_turn_shorthands(self.CUBE.get_solved_state(), short_hand_str.strip())
                state = data_moves_dict["data"][move_idx * test_n_cubes + cube_idx]
                self.assertTrue(np.all(state == test_state))

    def test_get_depth_1_lookup_of_state(self):
        init_state = np.expand_dims(
            self.CUBE.apply_turn_shorthands(self.CUBE.get_solved_state(), self.TEST_SHORTHAND_STR),
            0,
        ) # pad to have num_states = 1

        state = get_depth_1_lookup_of_state(init_state)
        self.assertEqual(state.shape[0], 12) # look up 12 moves

        for move_idx in range(12):
            shorthand_str = self.CUBE.TURN_IDX_TO_STR_SWAP_DICT[move_idx][0].replace("T", "U").replace("_PRIME", "'")
            test_state = self.CUBE.apply_turn_shorthands(init_state[0], shorthand_str)
            self.assertTrue(np.all(test_state == state[move_idx]))

    def test_find_best_move_and_value_from(self):
        test_n_cubes = 100
        init_move = "R" 
        reverse_move_idx = [i for i in range(12) if self.CUBE.TURN_IDX_TO_STR_SWAP_DICT[i][0] == (init_move + "_PRIME")][0]

        init_state = np.stack(
            [self.CUBE.get_solved_state()] * (test_n_cubes // 2)
            + [
                self.CUBE.apply_turn_shorthands(
                    self.CUBE.get_solved_state(),
                    init_move
                )
            ] * (test_n_cubes // 2),
            axis=0
        ) # (test_n_cubes, 54, 6)

        # test multiple patterns of unique weights assign to the states
        generator = np.random.default_rng(seed=131)
        for _ in range(10):
            eval_values = generator.random(size=(test_n_cubes, 12))
            def __evaluate_fn(state_values: np.ndarray):
                return eval_values.reshape((-1,))
            best_moves, best_values = find_best_move_and_value_from(
                init_state, evaluate_fn=__evaluate_fn
            ) 

            # check in aggregate
            reward_values = eval_values - 1
            reward_values[test_n_cubes//2:, reverse_move_idx] += 2
            test_moves = reward_values.argmax(axis=1)
            test_values = reward_values.max(axis=1)
            self.assertTrue(np.all(best_moves == test_moves))
            self.assertTrue(np.all(best_values == test_values))
            self.assertTrue(np.all(best_moves[test_n_cubes//2:] == reverse_move_idx))

            # manual loop
            for i, state_row, label_p, label_v in zip(
                range(len(init_state)), init_state, best_moves, best_values
            ):
                batched_lookup = get_depth_1_lookup_of_state(np.expand_dims(state_row, 0))
                batched_vs = __evaluate_fn(batched_lookup)[12*i:12*(i+1)]
                calc_v = batched_vs - 1                
                if i >= test_n_cubes // 2:
                    calc_v[reverse_move_idx] += 2
                self.assertTrue(np.all(batched_vs == eval_values[i]))
                self.assertTrue(label_v == calc_v.max(axis=0))
                self.assertTrue(label_p == calc_v.argmax(axis=0))
