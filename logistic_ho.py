import numpy as np
import jax.numpy as jnp
from jax import jacfwd
from sklearn.linear_model import LogisticRegression
from optim import newton


def loss_logistic(X, y, alpha, beta):
    n_samples, _ = X.shape
    logits = jnp.exp(X @ beta)
    logits = logits / (1 + logits)
    return -1 / n_samples * jnp.sum(
        y * jnp.log(logits) + (1 - y) * jnp.log(1 - logits)
    ) + jnp.exp(alpha) / 2 * jnp.sum(beta**2)


def grad_logistic(X, y, alpha, beta):
    """Gradient of the logistic loss function.
    Args:
        X: Design matrix.
        y: Response vector.
        alpha: Regularization parameter.
        beta: Coefficient vector where the logistic is evaluated.
    """
    n_samples, _ = X.shape
    logits = jnp.exp(X @ beta)
    logits = logits / (1 + logits)
    return 1 / n_samples * X.T @ (logits - y) + jnp.exp(alpha) * beta


def hessian_logistic(X, y, alpha, beta):
    """Hessian of the logistic loss function.
    Args:
        X: Design matrix.
        y: Response vector.
        alpha: Regularization parameter.
        beta: Coefficient vector where the logistic is evaluated.
    """
    n_samples, _ = X.shape
    logits = jnp.exp(X @ beta)
    logits = logits / (1 + logits)
    return 1 / n_samples * X.T @ jnp.diag(logits * (1 - logits)) @ X + jnp.exp(alpha) * jnp.eye(
        beta.size
    )


def newton_logistic(X, y, alpha, beta0, max_iter=50, solver="local", with_grad=False):
    """Newton's method for logistic regression.
    Args:
        X: Design matrix.
        y: Response vector.
        alpha: Regularization parameter.
        beta0: Initial coefficient vector.
        max_iter: Maximum number of iterations.
    """

    def grad_f(beta):
        return grad_logistic(X, y, alpha, beta)

    def hess_f(beta):
        return hessian_logistic(X, y, alpha, beta)

    if solver == "local":
        hbeta = newton(grad_f, hess_f, beta0, max_iter=max_iter)
    else:
        clf = LogisticRegression(
            solver=solver, C=1. / (np.exp(alpha) * X.shape[0]),
            penalty='l2', fit_intercept=False, tol=1e-15, max_iter=max_iter
        )
        clf.fit(X, y)
        hbeta = clf.coef_.flatten()

    if with_grad:
        jac_newton = jacfwd(newton_logistic, argnums=2)  # diff wrt alpha
        one_step_jac = jac_newton(X, y, alpha, hbeta, max_iter=1)
        return hbeta, one_step_jac
    else:
        return hbeta


def logistic_parameter_selection(
        X_train, X_val, y_train, y_val, alpha0, rho, max_iter=10, retall=False, solver="local"
):
    alpha = alpha0
    hbeta = jnp.zeros(X_train.shape[1])
    if retall:
        alphas = np.zeros(max_iter + 1)
        alphas[0] = alpha0
        losses = np.zeros(max_iter + 1)
        losses[0] = loss_logistic(X_val, y_val, alpha, hbeta)
    for i in range(max_iter):
        hbeta, hjac = newton_logistic(X_train, y_train, alpha, hbeta, with_grad=True, solver=solver)
        grad_outer = (grad_logistic(X_val, y_val, -jnp.inf, hbeta).T @ hjac.T).T
        alpha = alpha - rho * grad_outer
        if retall:
            alphas[i + 1] = alpha
            losses[i + 1] = loss_logistic(X_val, y_val, alpha, hbeta)
    if retall:
        return alpha, alphas, losses
    else:
        return alpha
