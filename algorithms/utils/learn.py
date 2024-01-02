"""Interface for learning in extensive-form games
"""

from typing import Callable

import numpy as np
import pyspiel

def main(
    game: pyspiel.Game,
    players: list[Callable[[list[float], list[int]], int]],
    num_episodes: int,
    fn: str,
):