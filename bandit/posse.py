"""
A gang of bandit agents for easily performing testing en masse.
"""

from typing import List, Type, Union

import numpy as np

from bandit.bandit import BaseBandit
from bandit.environment import Environment


class Posse:
    """
    A posse of bandits that all sample the same environment
    for the same number of steps.

    Args:
        environment (Environment): the environment that the bandits sample
        bandit_class (Type[BaseBandit]): the kind of bandit to create
        n_bandits (int): the number of bandits to create
        bandit_kwargs (dict): dictionary of arguments to pass to the bandits
    """

    def __init__(
        self,
        environment: Environment,
        bandit_class: Type[BaseBandit],
        n_bandits: int,
        **bandit_kwargs,
    ):
        self.environment: Environment = environment
        self.n_bandits: int = n_bandits
        self.bandits: List[Type[BaseBandit]] = [
            bandit_class(self.environment, **bandit_kwargs)
            for _ in range(n_bandits)
        ]
        self._n_actions_taken = 0

    def take_actions(self, n_actions: int) -> None:
        """
        Take `n_actions` actions for each bandit in the posse.

        Args:
            n_actions (int): number of actions to take
        """
        for _ in range(n_actions):
            for b in self.bandits:
                b.action()
        self._n_actions_taken += n_actions
        self.reward_histories = np.array([[]])
        self.choice_histories = np.array([[]])

    def __len__(self) -> int:
        return self._n_actions_taken

    @property
    def n_actions_taken(self) -> int:
        return self._n_actions_taken

    @property
    def len_env(self) -> int:
        return len(self.environment)

    @property
    def n_rewards(self) -> int:
        return self.len_env

    def _update_histories(self) -> None:
        """

        """
        self.reward_histories = np.array(
            [b.reward_history for b in self.bandits]
        )
        self.choice_histories = np.array(
            [b.choice_history for b in self.bandits]
        )

    def mean_reward(self) -> np.ndarray:
        """
        Average reward at each time computed over all bandits.
        """
        if self.n_actions_taken > len(self.reward_histories[0]):
            self._update_histories()
        return np.mean(self.reward_histories, axis=0)

    def var_reward(self) -> np.ndarray:
        """
        Variance at each time of the reward computed over all bandits.
        """
        if self.n_actions_taken > len(self.reward_histories[0]):
            self._update_histories
        return np.var(self.reward_histories, axis=0)

    def mean_best_choice(
        self, best_choice: Union[int, Union[List, np.ndarray]],
    ) -> np.ndarray:
        """
        Average of the best choice at each time computed over all bandits.

        Args:
            best_choice (Union[int, List[int], np.ndarray]): if int, the
                best choice for all times. If list of `np.ndarray` then
                the best choice at each time step.
        """
        if self.n_actions_taken > len(self.reward_histories[0]):
            self._update_histories()

        if type(best_choice) in [list, np.ndarray]:
            msg = "len(best_choices) must equal choice history of the bandits"
            assert len(best_choice) == len(self.choice_histories[0]), msg
            where_best = self.choice_histories == np.asarray(
                best_choice, dtype=np.int32
            )
        elif isinstance(best_choice, int):
            where_best = self.choice_histories == best_choice
        else:
            msg = f"best_choice must be int, list, np.ndarray but {type(best_choice)} provided"  # noqa: E501
            raise TypeError(msg)

        return np.mean(where_best, axis=0)

    def var_best_choice(
        self, best_choice: Union[int, Union[List, np.ndarray]],
    ) -> np.ndarray:
        """
        Average of the best choice at each time computed over all bandits.

        Args:
            best_choice (Union[int, List[int], np.ndarray]): if int, the
                best choice for all times. If list of `np.ndarray` then
                the best choice at each time step.
        """
        if self.n_actions_taken > len(self.reward_histories[0]):
            self._update_histories()

        if type(best_choice) in [list, np.ndarray]:
            msg = "len(best_choices) must equal choice history of the bandits"
            assert len(best_choice) == len(self.choice_histories[0]), msg
            where_best = self.choice_histories == np.asarray(
                best_choice, dtype=np.int32
            )
        elif isinstance(best_choice, int):
            where_best = self.choice_histories == best_choice
        else:
            msg = f"best_choice must be int, list, np.ndarray but {type(best_choice)} provided"  # noqa: E501
            raise TypeError(msg)

        return np.var(where_best, axis=0)
