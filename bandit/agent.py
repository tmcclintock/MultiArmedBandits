"""
Bandit agents that implement various strategies.
"""

from abc import ABC, abstractmethod

from bandit.environment import Environment

import numpy as np


class BaseBandit(ABC):
    """
    Base class for all bandit agents.
    """

    def __init__(self, environment: Environment):
        self.environment = environment

    @abstractmethod
    def choose_action(self, *args, **kwargs):
        return 0

    def action(self, i: int = None):
        a = self.choose_action() if i is None else 0
        return self.environment.action(a)


class RandomBandit(BaseBandit):
    """
    A totally random bandit with no strategy.
    Actions are selected randomly.
    """

    def choose_action(self, *args, **kwargs):
        """
        Choose a random action.
        """
        return np.random.randint(0, len(self.environment))
