"""
Bandit agents that implement various strategies.
"""

from typing import List, Tuple

from abc import ABC, abstractmethod

from bandit.environment import Environment

import numpy as np


class BaseBandit(ABC):
    """
    Base class for all bandit agents.
    """

    def __init__(self, environment: Environment, values: List[float] = None):
        self.environment = environment
        if values is None:
            self.values = [0.0] * len(self.environment)
        else:
            self.values = values
        self.reward_history = []
        self.choice_history = []

    @abstractmethod
    def choose_action(self, *args, **kwargs) -> int:
        return 0

    def action(self, i: int = None) -> float:
        choice = self.choose_action() if i is None else 0
        self.choice_history.append(choice)
        reward = self.environment.action(choice)
        self.reward_history.append(reward)
        return reward

    @property
    def history(self) -> Tuple[List, List]:
        return (self.reward_history, self.choice_history)


class RandomBandit(BaseBandit):
    """
    A totally random bandit with no strategy.
    Actions are selected randomly.
    """

    def choose_action(self, *args, **kwargs) -> int:
        """
        Choose a random action.
        """
        return np.random.randint(0, len(self.environment))
