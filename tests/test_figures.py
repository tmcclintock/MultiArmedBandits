"""
Test of the figure routines.
"""

from bandit.bandit import RandomBandit
from bandit.environment import Environment
from bandit.figures import plot_average_rewards, plot_reward_distributions
from bandit.reward import GaussianReward, PoissonReward

import matplotlib as mpl

from unittest import TestCase
import pytest


class TestFigures(TestCase):
    def setUp(self):
        super().setUp()
        N = 5
        self.n_rewards = N
        self.env = Environment([GaussianReward(i) for i in range(N)])
        self.env2 = Environment([PoissonReward(i + 1) for i in range(N)])
        N_bandits = 10
        N_steps = 10
        self.bandits = [RandomBandit(self.env) for _ in range(N_bandits)]
        for i in range(N_bandits):
            for _ in range(N_steps):
                self.bandits[i].action()

    @pytest.mark.slow
    def test_plot_reward_distributions(self):
        # Test a continuous dist.
        fig, ax = plot_reward_distributions(self.env.rewards)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)
        # Test a discrete dist.
        fig, ax = plot_reward_distributions(self.env2.rewards)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)

    @pytest.mark.slow
    def test_plot_reward_distributions_with_axis(self):
        fig, axis = mpl.pyplot.subplots()
        fig, ax = plot_reward_distributions(self.env.rewards, axis=axis)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)
        assert ax == axis

    @pytest.mark.slow
    def test_plot_average_rewards(self):
        reward_histories = [b.history[0] for b in self.bandits]
        fig, ax = plot_average_rewards(reward_histories)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)


if __name__ == "__main__":
    tf = TestFigures()
    tf.setUp()
    tf.test_plot_reward_distributions()
    tf.test_plot_average_rewards()
    mpl.pyplot.show()
