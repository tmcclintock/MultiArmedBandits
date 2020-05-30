"""
Tests of cells.
"""

from unittest import TestCase

from bandit.cell import Cell
from bandit.bandit import EpsGreedyBandit, GreedyBandit
from bandit.environment import Environment
from bandit.reward import GaussianReward


class CellTestCase(TestCase):
    def setUp(self):
        super().setUp()
        Nr = 5
        self.n_rewards = Nr
        self.env = Environment([GaussianReward() for _ in range(Nr)])

    def test_smoke(self):
        cell = Cell(self.env, GreedyBandit, n_bandits=20)
        assert isinstance(cell, Cell)

    def test_numbers(self):
        N_bandits = 20
        cell = Cell(self.env, GreedyBandit, n_bandits=N_bandits)
        assert len(cell.bandits) == N_bandits
        assert cell.n_actions_taken == 0
        assert cell.len_env == len(self.env)
        assert cell.n_rewards == self.n_rewards

    def test_actions(self):
        N_bandits = 20
        cell = Cell(self.env, GreedyBandit, n_bandits=N_bandits)
        N_actions = 2
        cell.take_actions(N_actions)
        assert cell.n_actions_taken == 2
        assert len(cell.bandits[0].choice_history) == 2

    def test_bandit_kwargs(self):
        N_bandits = 20
        eps = 0.1
        cell = Cell(self.env, EpsGreedyBandit, n_bandits=N_bandits, eps=eps)
        assert cell.bandits[0].eps == eps
