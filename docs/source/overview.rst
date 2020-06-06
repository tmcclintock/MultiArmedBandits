Overview
========

Introduction to bandit problems
-------------------------------

Bandit problems are a class of machine learning problems
concerned with attempting to optimize return given
a fixed set of choices (actions) we can take
from a fixed but possibly non-stationary state. It is
specialized case of reinforcement learning where there
is only a single state.

For this introduction I will consider only stationary
bandit problems.

In brief, suppose our bandit is selecting actions :math:`a`
that result in a reward drawn from a random distribution
:math:`r\sim P(R|a)`. This distribution can be continuous,
discrete, or mixed (or even a deterministic
single value :math:`P(R|a) =\delta(R-r)`). We say that
this distribution is controlled by the environment :math:`\mathcal{E}`.
The goal of the bandit is to try to maximize the sum of the
rewards (called the total
return :math:`G`) given :math:`N` opportunities to receive
a reward from the environment.

In standard bandit problems, there are a finite number of
discrete number :math:`n` of actions :math:`a_i\in\{a_1,a_2,...,a_n \}=\mathcal{A}`.

Thus, the total return can be written as

.. math::

   G = \sum_{i=1}^N r_i\,,\ r_i\sim P(R|a\in\mathcal{A})\,.

Assuming :math:`N` is large, this means that we can obtain
an optimal return by selecting an action that yields the highest
exected return value

.. math::

   \hat{a} = \underset{a}{\operatorname{argmax}}\mathbb{E}[R|a]\,.

The purpose of bandit **algorithms** are to try to learn from the
environment which action is optimal.

This repository is created to help explore this question in
different contexts, or to run bandit experiments. You can find
modular, fully tested implementations of **rewards**, **environments**,
**bandits**, and a helpful construction called a **posse** or
a collection of bandits to run an experiment with. In addition,
helpful standard plotting routines for investigating
bandit statistics are implemented in the **figures** methods.

Read on to see advanced bandit problems solved using this package,
as well as references for further learning.
