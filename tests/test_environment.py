"""
Tests of the environment object.
"""

import numpy as np
import pytest

from unittest import TestCase

from bandit.environment import Environment
from bandit.reward import GaussianReward, PoissonReward


class TestEnvironment(TestCase):
    def test_gaussian_rewards(self):
        n = 10
        rewards = []
        ms = np.linspace(-1, 1, n)
        vs = np.linspace(0.1, 3, n)
        for m, v in zip(ms, vs):
            rewards.append(GaussianReward(m, v))
        e = Environment(rewards)
        assert len(e) == n
        er = e.expected_rewards()
        for i, r in enumerate(er):
            assert r == rewards[i].expected_reward()

    def test_poisson_rewards(self):
        n = 10
        rewards = []
        mus = np.linspace(1, 5, n)
        for mu in mus:
            rewards.append(PoissonReward(mu))
        e = Environment(rewards)
        assert len(e) == n
        er = e.expected_rewards()
        for i, r in enumerate(er):
            assert r == rewards[i].expected_reward()

    def test_action(self):
        rewards = [PoissonReward(mu) for mu in np.random.rand(10) + 1]
        e = Environment(rewards)
        with pytest.raises(AssertionError):
            e.action(11)
        r = e.action(0)
        print(r)
        assert type(r) in [float, int]

    def test_moments(self):
        rewards = [PoissonReward(mu) for mu in [4] * 5]
        e = Environment(rewards)
        moments = e.moments()
        assert np.all(moments == 4)
