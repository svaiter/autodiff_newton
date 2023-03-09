import numpy as np
import matplotlib.pyplot as plt


def plot_classification(X, y, betas, s=20, titles=None):
    """ Plot the data points X and and classification boundary (only in 2D).
    Args:
        X: design, the first two columns are ploted, the third column is full of ones
        y: labels in {0, 1}
        betas: estimated parameters, third entry is the intercept.
        s: optional size
        titles: optional titles
    """
    cplot = y
    n_betas = len(betas)
    fig, axs = plt.subplots(1, n_betas, sharex=True, sharey=True)

    for i in range(n_betas):
        beta = betas[i]
        axs[i].scatter(X[:, 0], X[:, 1], s=s, c=cplot)

        a1 = np.min(X[:, 0])
        a2 = np.max(X[:, 0])
        b1 = -(beta[0] * a1 + beta[2]) / beta[1]
        b2 = -(beta[0] * a2 + beta[2]) / beta[1]
        axs[i].plot([a1, a2], [b1, b2])
        if titles:
            axs[i].set_title(titles[i])

    return fig, axs
