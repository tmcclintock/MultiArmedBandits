"""
Test of the figure routines.
"""

from bandit.environment import Environment
from bandit.figures import plot_reward_distributions
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
    def test_plot_with_axis(self):
        fig, axis = mpl.pyplot.subplots()
        fig, ax = plot_reward_distributions(self.env.rewards, axis=axis)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)
        assert ax == axis


if __name__ == "__main__":
    tf = TestFigures()
    tf.setUp()
    tf.test_plot_reward_distributions()
    mpl.pyplot.show()
