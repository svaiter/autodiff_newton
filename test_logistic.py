import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from utils import plot_classification
from logistic_ho import newton_logistic, logistic_parameter_selection

key = jax.random.PRNGKey(123)
rng = default_rng(seed=123)

setting = "random"

if setting == "2dcloud":
    n = 1000
    n2 = n // 2
    p = 2

    # Initial estimate for the logisitic regression parameters
    beta = jnp.zeros(p+1) + 0.1

    # Labels
    y = np.zeros(n)
    y[n2:n] = 1

    # Design matrix, standard gaussian, entries corresponding to label 1 are shifted.
    X = rng.standard_normal(size=(n, p + 1))
    X[n2:n, :] = X[n2:n, :] + 2.0
    X[0:n2, :] = X[0:n2, :] + 1.0
    X[:, p] = 1.0
elif setting == "random":
    n = 100
    p = 20
    beta = jnp.zeros(p)
    X, y = make_classification(n_samples=n, n_features=p)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = (
    jnp.array(X_train),
    jnp.array(X_val),
    jnp.array(y_train),
    jnp.array(y_val),
)

# Regularization
alpha0 = 1.0
rho = 0.05
alpha = logistic_parameter_selection(X_train, X_val, y_train, y_val, rho, alpha0, max_iter=100)
beta_final = newton_logistic(X_train, y_train, alpha, beta, max_iter=50)


# Plot

# fig_after, ax_after = plot_classification(
#     X_val, y_val, [beta, beta_final], titles=["Initialization", "After training"]
# )

# plt.show()
