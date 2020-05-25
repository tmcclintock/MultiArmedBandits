"""
Tests of bandits.
"""

import numpy as np
from unittest import TestCase

from bandit.bandit import EpsGreedyBandit, GreedyBandit, RandomBandit
from bandit.environment import Environment
from bandit.reward import GaussianReward


class BanditTestCase(TestCase):
    def setUp(self):
        super().setUp()
        N = 5
        self.n_rewards = N
        self.env = Environment([GaussianReward() for _ in range(N)])


class TestRandomBandit(BanditTestCase):
    def test_smoke(self):
        b = RandomBandit(self.env)
        assert isinstance(b, RandomBandit)
        assert isinstance(b.environment, Environment)
        assert len(b.environment) == self.n_rewards
        assert b.reward_history == []
        assert b.choice_history == []

    def test_history(self):
        b = RandomBandit(self.env)
        rh, ch = b.history
        assert rh == []
        assert ch == []
        assert len(b) == 0
        _ = b.action()
        rh, ch = b.history
        assert len(rh) == 1
        assert len(ch) == 1
        assert isinstance(rh[0], float)
        assert isinstance(ch[0], int)
        assert len(b) == 1
        for _ in range(99):
            _ = b.action()
        rh, ch = b.history
        assert len(rh) == 100
        assert len(ch) == 100
        assert isinstance(rh[99], float)
        assert isinstance(ch[99], int)
        assert len(b) == 100

    def test_choose_action(self):
        b = RandomBandit(self.env)
        assert isinstance(b.choose_action(), int)

    def test_action(self):
        b = RandomBandit(self.env)
        a = b.action()
        assert isinstance(b.action(), float)
        a2 = b.action()
        assert a != a2  # unless we get very unlucky

    def test_values(self):
        b = RandomBandit(self.env)
        assert b.values == [0.0] * len(self.env)
        b = RandomBandit(self.env, values=[1.0] * len(self.env))
        assert b.values == [1.0] * len(self.env)


class TestGreedyBandit(BanditTestCase):
    def test_choose_action(self):
        b = GreedyBandit(self.env)
        assert np.issubdtype(b.choose_action(), np.integer)
        assert np.issubdtype(b.action(), np.floating)


class TestEpsGreedyBandit(BanditTestCase):
    def test_choose_action(self):
        eps = 1.0
        b = EpsGreedyBandit(self.env, eps)
        assert hasattr(b, "eps")
        assert b.eps == eps
        assert np.issubdtype(b.choose_action(), np.integer)
        assert np.issubdtype(b.action(), np.floating)
        eps = 0.0
        b = EpsGreedyBandit(self.env, eps)
        assert hasattr(b, "eps")
        assert b.eps == eps
        assert np.issubdtype(b.choose_action(), np.integer)
        assert np.issubdtype(b.action(), np.floating)
        eps = 0.1
        b = EpsGreedyBandit(self.env, eps)
        assert hasattr(b, "eps")
        assert b.eps == eps
        assert np.issubdtype(b.choose_action(), np.integer)
        assert np.issubdtype(b.action(), np.floating)
