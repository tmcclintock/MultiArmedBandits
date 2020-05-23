"""
Classes for the environment and the reward model.
"""

from typing import Callable

import numpy as np
import scipy.stats as ss

from abc import ABC, abstractmethod


class BaseReward(ABC):
    """
    Base class for rewards

    Args:
        dist (Callable): a random variable distribution that has an `rvs`
            method that returns a reward
    """

    def __init__(self, dist: Callable):
        assert hasattr(dist, "rvs"), "distribution must have rvs() method"
        assert hasattr(dist, "stats"), "distribution must have a stats method"
        self.dist = dist

    @abstractmethod
    def get_reward(self):
        return self.dist.rvs()

    @abstractmethod
    def expected_reward(self):
        return self.dist.stats("m")

    def moments(self):
        return self.dist.stats("mv")


class GaussianReward(BaseReward):
    """
    A Gaussian random variable as a reward.

    Args:
        mean (float): mean of the Gaussian reward
        var (float): variance of the Gaussian reward; must be positive
    """

    def __init__(self, mean: float = 0, var: float = 1):
        assert var > 0, "variance must be positive"
        super().__init__(ss.norm(loc=mean, scale=np.sqrt(var)))

    def get_reward(self):
        return super().get_reward()

    def expected_reward(self):
        return super().expected_reward()


class PoissonReward(BaseReward):
    """
    Poisson random variable reward.

    Args:
        mu (float): rate parameter (mean and var)
        loc (float): constant shift
    """

    def __init__(self, mu: float = 1, loc: float = 0):
        assert mu > 0, "poisson rate must be positive"
        super().__init__(ss.poisson(mu=mu, loc=loc))

    def get_reward(self):
        return super().get_reward()

    def expected_reward(self):
        return super().expected_reward()
