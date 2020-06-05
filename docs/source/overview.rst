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

   G = \sum_{i=1}^N r_i\,,\ r_i\sim P(R|a\in\mathcal{A})
