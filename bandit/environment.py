"""
Single-state environments that contain
"""

from typing import List

from bandit.reward import GaussianReward, PoissonReward

import numpy as np


class Environment:
    """
    A single-state environment that contains a list of rewards for actions.
    """

    def __init__(self, rewards: List):
        for r in rewards:
            assert type(r) in [GaussianReward, PoissonReward], "invalid reward"
        self.rewards = rewards

    def __len__(self):
        return len(self.rewards)

    def action(self, i: int):
        assert i > -1
        assert i < self.__len__()
        return self.rewards[i].get_reward()

    def expected_rewards(self):
        return np.array([r.expected_reward() for r in self.rewards])

    def moments(self):
        return np.array([r.moments() for r in self.rewards])
