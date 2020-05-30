# MultiArmedBandits [![Build Status](https://travis-ci.com/tmcclintock/MultiArmedBandits.svg?branch=master)](https://travis-ci.com/tmcclintock/MultiArmedBandits)

A package to build and test bandit algorithms.

Bandit algorithms learn how to balance exploitation with exploration when
attempting to maximize rewards obtained from a single-state (but possibly
non-stationary) environment. This repository is for building and experimenting
with various bandit algorithms.

## Installation

Install by running these two commands to install the requirements and this
package
```bash
pip install -r requirements.txt
pip install .
```
If you use `conda` then you can create the environment for this package
(called `mab`) from the `environment.yml` file with
```bash
conda env create -f environment.yml
```
and then install with
```bash
pip install .
```
Once installed, test your installation by running
```bash
pytest
```
by default you should have all passing tests and some skipped tests.

## Background

Bandit problems are situations in which one is faced with a series of
repeatable choices that yield a (possibly random) reward. The goal is
to develop a strategy to maximize the rewards obtained using some
kind of strategy or algorithm. Each time a reward is obtained, one
is faced with the same set of choices, hence it is a single-state
learning problem with a set of actions and rewards.

The vocabulary used in this package is that the agent is called a `bandit`,
which takes `actions` to receive `rewards`, which is handled by the
`environment`.

This package contains a modular, tested framework for developing, deploying
and analyzing bandit algorithms. This includes easily modified classes
for `rewards` to incorporate, e.g., non-stationarity. New bandit algorithms
are easily created, requiring only the `choose_action` function to be
written.

This package contains features to facilitate experiments, including
the `posse` object to represent a collection of bandits as well as premade
functions for plotting.

## Usage

Suppose you want to experiment with a greedy bandit in an environment
with ten Gaussian (normal) distributed rewards, and you would like the
bandit to take one thousand actions in order to see how it performed. This
can be accomplished in the following short program:
```python
from bandit import *

rewards = [GaussienReward() for _ in range(10)]
environment = Environment(rewards)
bandit = GreedyBandit(environment)

for _ in range(1000):
    bandit.action()
```
the results of this experiment (reward and choice history) are stored in
`bandit.history` for easy analysis.