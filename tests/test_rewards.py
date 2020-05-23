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
