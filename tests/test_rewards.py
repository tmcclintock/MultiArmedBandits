"""
Tests of the rewards.
"""

import numpy as np

from unittest import TestCase

from bandit import GaussianReward, PoissonReward


class TestGaussianReward(TestCase):
    def test_smoke(self):
        _ = GaussianReward()
        assert True

    def test_stat_moments(self):
        gr = GaussianReward(mean=np.pi, var=np.e)
        # Moments
        m, v, s, k = gr.dist.stats(moments="mvsk")
        assert m == np.pi
        np.testing.assert_almost_equal(v, np.e)
        assert s == 0
        assert k == 0

    def test_expected_reward(self):
        gr = GaussianReward(mean=3)
        assert gr.expected_reward() == 3

    def test_moments(self):
        gr = GaussianReward(mean=3, var=np.pi)
        m, v = gr.moments()
        assert m == 3
        np.testing.assert_almost_equal(v, np.pi)


class TestPoissonReward(TestCase):
    def test_smoke(self):
        _ = PoissonReward()
        assert True

    def test_stat_moments(self):
        pr = PoissonReward(mu=np.pi)
        # Moments
        m, v = pr.dist.stats(moments="mv")
        assert m == np.pi
        assert v == np.pi

    def test_expected_reward(self):
        pr = PoissonReward(mu=3)
        assert pr.expected_reward() == 3

    def test_moments(self):
        pr = PoissonReward(mu=np.pi)
        m, v = pr.moments()
        assert m == np.pi
        assert v == np.pi
