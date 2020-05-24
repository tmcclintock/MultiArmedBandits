"""
Bandit agents that implement various strategies.
"""

from typing import List, Tuple, Union

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
        self._n = 0

    def __len__(self):
        return self._n

    @abstractmethod
    def choose_action(self, *args, **kwargs) -> int:
        return 0  # pragma: no cover

    def update_history_and_values(
        self, choice: int, reward: Union[float, int]
    ) -> None:
        """
        Update the histories and the value estimates. This base
        class assumes a sample mean estimate for the values.
        Different strategies require overwriting this function.

        Args:
            choice (int): choiec of action taken
            reward (Union[float, int]): reward recieved
        """
        self.values[choice] += float(reward - self.values[choice]) / (
            self._n + 1
        )
        self.choice_history.append(choice)
        self.reward_history.append(reward)
        self._n += 1
        return

    def action(self, i: int = None) -> float:
        """
        Take an action.
        Args:
            i (int): action to take

        Returns:
            (float) reward of the taken action
        """
        choice = self.choose_action() if i is None else 0
        reward = self.environment.action(choice)
        self.update_history_and_values(choice, reward)
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

        Returns:
            (int) action choice
        """
        return np.random.randint(0, len(self.environment))


class GreedyBandit(BaseBandit):
    """
    Greedy bandit that always selects the optimally valued
    action.
    """

    def choose_action(self, *args, **kwargs) -> int:
        """
        Choose the action with the highest value.
        In case of any ties, return a random selection.

        Returns:
            (int) action choice
        """
        return np.random.choice(
            np.where(self.values == np.max(self.values))[0]
        )
