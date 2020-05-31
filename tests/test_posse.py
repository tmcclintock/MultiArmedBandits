"""
Tests of posses.
"""

import numpy as np
import pytest
from unittest import TestCase

from bandit.posse import Posse
from bandit.bandit import EpsGreedyBandit, GreedyBandit
from bandit.environment import Environment
from bandit.reward import GaussianReward


class PosseTestCase(TestCase):
    def setUp(self):
        super().setUp()
        Nr = 5
        self.n_rewards = Nr
        self.env = Environment([GaussianReward() for _ in range(Nr)])

    def test_smoke(self):
        posse = Posse(self.env, GreedyBandit, n_bandits=20)
        assert isinstance(posse, Posse)

    def test_numbers(self):
        N_bandits = 20
        posse = Posse(self.env, GreedyBandit, n_bandits=N_bandits)
        assert len(posse.bandits) == N_bandits
        assert posse.n_actions_taken == 0
        assert posse.len_env == len(self.env)
        assert posse.n_rewards == self.n_rewards

    def test_actions(self):
        N_bandits = 20
        posse = Posse(self.env, GreedyBandit, n_bandits=N_bandits)
        N_actions = 2
        posse.take_actions(N_actions)
        assert posse.n_actions_taken == 2
        assert len(posse.bandits[0].choice_history) == 2

    def test_bandit_kwargs(self):
        N_bandits = 20
        eps = 0.1
        posse = Posse(self.env, EpsGreedyBandit, n_bandits=N_bandits, eps=eps)
        assert posse.bandits[0].eps == eps

    def test_mean_var_reward(self):
        N_bandits = 20
        posse = Posse(self.env, GreedyBandit, n_bandits=N_bandits)
        N_actions = 100
        posse.take_actions(N_actions)
        assert posse.mean_reward().shape == (100,)
        assert posse.var_reward().shape == (100,)

    def test_mean_best_choice(self):
        N_bandits = 20
        posse = Posse(self.env, GreedyBandit, n_bandits=N_bandits)
        N_actions = 100
        posse.take_actions(N_actions)
        bc = np.zeros(100)
        assert posse.mean_best_choice(bc).shape == (100,)
        assert posse.mean_best_choice(0).shape == (100,)
        assert posse.mean_best_choice(bc.tolist()).shape == (100,)
        with pytest.raises(TypeError):
            posse.mean_best_choice(3.1415)
