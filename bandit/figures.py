"""
Visualizations for bandit experiments.
"""

from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import rv_discrete, rv_continuous


def plot_reward_distributions(
    rewards: List, axis: "mpl.axes.Axes" = None
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Create violin plots of the reward distributions.

    Args:
        rewards (List[Rewards]): rewards to make distributions of
        axis (mpl.axes.Axes) axis to use for plotting, default `None`
    """
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = plt.gcf()

    # Horizontal positions of the centers of the violins
    positions = np.arange(0, len(rewards))
    axis.set_xlim(positions.min() - 0.5, positions.max() + 0.5)

    # Loop over all rewards and draw the violin
    for i, r in zip(positions, rewards):
        interval = r.dist.interval(0.99999)  # 5-sigma

        # Handle continuous vs discrete cases differently
        if hasattr(r.dist, "dist"):

            if isinstance(r.dist.dist, rv_discrete):
                x = np.arange(min(interval), max(interval) + 1)
                y = r.dist.pmf(x)
                scale = 0.4 / y.max()
                for xi, yi in zip(x, y):
                    plt.plot(
                        [i - yi * scale, i - yi * scale],
                        [xi, xi + 1],
                        color="gray",
                        alpha=0.5,
                    )
                    plt.plot(
                        [i + yi * scale, i + yi * scale],
                        [xi, xi + 1],
                        color="gray",
                        alpha=0.5,
                    )
            elif isinstance(r.dist.dist, rv_continuous):
                x = np.linspace(min(interval), max(interval), 100)
                y = r.dist.pdf(x)
                scale = 0.4 / y.max()
                plt.plot(i - y * scale, x, color="gray", alpha=0.5)
                plt.plot(i + y * scale, x, color="gray", alpha=0.5)
            else:  # need to do random draws
                raise NotImplementedError(
                    "only scipy.stats distributions supported"
                )  # pragma: no cover
        else:
            raise NotImplementedError(
                "only scipy.stats distributions supported"
            )  # pragma: no cover

    return fig, axis
