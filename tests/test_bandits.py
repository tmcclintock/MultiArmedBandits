"""
Tests of bandits.
"""

from unittest import TestCase

from bandit.bandit import RandomBandit
from bandit.environment import Environment
from bandit.reward import GaussianReward


class TestRandomBandit(TestCase):
    def setUp(self):
        super().setUp()
        N = 5
        self.n_rewards = N
        self.env = Environment([GaussianReward() for _ in range(N)])

    def test_smoke(self):
        b = RandomBandit(self.env)
        assert isinstance(b, RandomBandit)
        assert isinstance(b.environment, Environment)
        assert len(b.environment) == self.n_rewards

    def test_choose_action(self):
        b = RandomBandit(self.env)
        assert isinstance(b.choose_action(), int)

    def test_action(self):
        b = RandomBandit(self.env)
        a = b.action()
        assert isinstance(b.action(), float)
        a2 = b.action()
        assert a != a2  # unless we get very unlucky
