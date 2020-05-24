"""
Single-state environments that contain
"""

from typing import List, Union

import numpy as np

from bandit.reward import GaussianReward, PoissonReward


class Environment:
    """
    A single-state environment that contains a list of rewards for actions.
    """

    def __init__(self, rewards: List):
        for r in rewards:
            assert type(r) in [GaussianReward, PoissonReward], "invalid reward"
        self.rewards = rewards

    def __len__(self) -> int:
        return len(self.rewards)

    def action(self, i: int) -> Union[float, int]:
        assert i > -1
        assert i < self.__len__()
        return self.rewards[i].get_reward()

    def expected_rewards(self) -> float:
        return np.array([r.expected_reward() for r in self.rewards])

    def moments(self, kind: str = "mv") -> List[float]:
        return np.array([r.moments(kind) for r in self.rewards])
