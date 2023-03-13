import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_iris
from libsvmdata.datasets import fetch_libsvm

from utils import plot_classification
from logistic_ho import newton_logistic, logistic_parameter_selection, loss_logistic

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
    n = 800
    p = 50
    beta = jnp.zeros(p)
    X, y = make_classification(n_samples=n, n_features=p)
elif setting == "libsvm":
    X, y = fetch_libsvm('rcv1.binary')
    X = X[:, :100].todense()
    n, p = X.shape
    beta = jnp.zeros(p)

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
alpha0 = 0.0
rho = 0.1
alpha, alphas, losses = logistic_parameter_selection(X_train, X_val, y_train, y_val, alpha0, rho, max_iter=10000, retall=True)
beta_final = newton_logistic(X_train, y_train, alpha, beta, max_iter=50)

fig, ax = plt.subplots(1,2)
ax[0].plot(losses)
ax[1].plot(alphas)
plt.show()

## Optuna

# def objective(alpha):
#     alpha = alpha.suggest_float('alpha', -10, 10.0)
#     beta = newton_logistic(X_train, y_train, alpha, np.zeros(X_train.shape[1]))
#     return loss_logistic(X_val, y_val, -jnp.inf, beta)

# try:
#     import optuna
#     study = optuna.create_study()
#     study.optimize(objective, n_trials=100)
#     study.best_params
# except ImportError:
#     print("optuna not available")

# Plot

# fig_after, ax_after = plot_classification(
#     X_val, y_val, [beta, beta_final], titles=["Initialization", "After training"]
# )

# plt.show()
