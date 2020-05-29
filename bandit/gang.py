"""
A gang of bandit agents for easily performing testing en masse.
"""

from typing import List, Type

from bandit.bandit import BaseBandit
from bandit.environment import Environment


class Gang:
    """
    A gang of bandits that all sample the same environment
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
        Take `n_actions` actions for each bandit in the gang.

        Args:
            n_actions (int): number of actions to take
        """
        for _ in range(n_actions):
            for b in self.bandits:
                b.action()
        self._n_actions_taken += n_actions

    def __len__(self) -> int:
        return self._nactions_taken

    @property
    def n_actions_taken(self) -> int:
        return self.n_actions_taken

    @property
    def len_env(self) -> int:
        return len(self.environment)

    @property
    def n_rewards(self) -> int:
        return self.len_env
