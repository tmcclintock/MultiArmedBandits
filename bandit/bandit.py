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
        self.n_selections = np.zeros(len(self.environment), dtype=np.int32)
        self.reward_history = []
        self.choice_history = []

    def __len__(self):
        return len(self.choice_history)

    @abstractmethod
    def choose_action(self, *args, **kwargs) -> int:
        pass  # pragma: no cover

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
        self.n_selections[choice] += 1
        self.values[choice] += float(reward - self.values[choice]) / (
            self.n_selections[choice]
        )
        self.choice_history.append(choice)
        self.reward_history.append(reward)

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


class CustomBandit(BaseBandit):
    """
    Wrapper around the `BaseBandit` for creating custom
    bandit subclasses.
    """

    def choose_action(self, *args, **kwargs) -> int:
        raise NotImplementedError


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


class EpsGreedyBandit(BaseBandit):
    """
    Epsilon-Greedy bandit, that makes a random choice
    100*episilon percent of the time for exploration
    and acts greedily the rest of the time.

    Args:
        eps (float): fraction of time taking exploratory actions
    """

    def __init__(
        self, environment: Environment, eps: float, values: List[float] = None
    ):
        super().__init__(environment, values)
        self.eps = eps

    def choose_action(self, *args, **kwargs) -> int:
        """
        Choose a random action `100*self.eps` percent of the time
        and otherwise take greedy actions.

        Returns:
            (int) action choice
        """
        if np.random.rand() < self.eps:  # random step
            return np.random.randint(len(self.environment), dtype=np.int32)
        else:  # greedy step
            return np.random.choice(
                np.where(self.values == np.max(self.values))[0]
            )
