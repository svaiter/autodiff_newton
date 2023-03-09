import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jacfwd

from utils import plot_classification
from optim import newton

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


def grad_logistic(X, y, alpha, beta):
    logits = jnp.exp(X @ beta)
    logits = logits / (1 + logits)
    return X.T @ (logits - y) + alpha * beta


def hessian_logistic(X, y, alpha, beta):
    logits = jnp.exp(X @ beta)
    logits = logits / (1 + logits)
    return (X * logits * (1 - logits)).T @ X + alpha * jnp.eye(beta.size)


# Regularization
alpha = 1.0


def newton_logistic(X, y, alpha, beta0, max_iter=50):
    def grad_f(beta):
        return grad_logistic(X, y, alpha, beta)

    def hess_f(beta):
        return hessian_logistic(X, y, alpha, beta)

    hbeta = newton(grad_f, hess_f, beta0, max_iter=max_iter)
    return hbeta


beta_final = newton_logistic(X, y, alpha, beta, max_iter=50)

# One step of Newton

jac_newton = jacfwd(newton_logistic, argnums=2)  # diff wrt alpha
one_step_jac = jac_newton(X, y, alpha, beta_final, max_iter=1)
np.testing.assert_array_almost_equal(
    one_step_jac
    + jnp.linalg.solve(hessian_logistic(X, y, alpha, beta_final), beta_final),
    np.zeros_like(beta_final),
)

# Plot

fig_after, ax_after = plot_classification(
    X, y, [beta, beta_final], titles=["Initialization", "After training"]
)

plt.show()
