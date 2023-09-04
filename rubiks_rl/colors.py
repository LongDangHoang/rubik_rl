import numpy as np

class Color:
    RED = 0
    BLUE = 1
    WHITE = 2
    ORANGE = 3
    YELLOW = 4
    GREEN = 5

    @classmethod
    def one_hot(cls, idx: int) -> np.ndarray:
        v = np.zeros(6)
        v[idx] = 1
        return v

    @classmethod
    def get_str_name(cls, idx: int) -> str:
        if idx == cls.RED:
            return "RED"
        if idx == cls.BLUE:
            return "BLUE"
        if idx == cls.WHITE:
            return "WHITE"
        if idx == cls.ORANGE:
            return "ORANGE"
        if idx == cls.YELLOW:
            return "YELLOW"
        if idx == cls.GREEN:
            return "GREEN"
