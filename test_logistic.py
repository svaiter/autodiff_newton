import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from utils import plot_classification
from logistic_ho import newton_logistic

key = jax.random.PRNGKey(123)
rng = default_rng(seed=123)

n = 1000
n2 = n // 2
p = 2

# Initial estimate for the logisitic regression parameters
beta = jnp.zeros((p + 1, 1)) + 0.1

# Labels
y = np.zeros((n, 1))
y[n2:n] = 1
y = jnp.array(y)

# Design matrix, standard gaussian, entries corresponding to label 1 are shifted.
X = rng.standard_normal(size=(n, p + 1))
X[n2:n, :] = X[n2:n, :] + 2.0
X[0:n2, :] = X[0:n2, :] + 1.0
X[:, p] = 1.0
X = jnp.array(X)

# Regularization
alpha = 1.0
beta_final, grad_final = newton_logistic(X, y, alpha, beta, max_iter=50, with_grad = True)


# Plot

fig_after, ax_after = plot_classification(
    X, y, [beta, beta_final], titles=["Initialization", "After training"]
)

plt.show()
