import jax.numpy as jnp
from jax import jacfwd
from optim import newton


def grad_logistic(X, y, alpha, beta):
    """Gradient of the logistic loss function.
    Args:
        X: Design matrix.
        y: Response vector.
        alpha: Regularization parameter.
        beta: Coefficient vector where the logistic is evaluated.
    """
    logits = jnp.exp(X @ beta)
    logits = logits / (1 + logits)
    return X.T @ (logits - y) + alpha * beta


def hessian_logistic(X, y, alpha, beta):
    """Hessian of the logistic loss function.
    Args:
        X: Design matrix.
        y: Response vector.
        alpha: Regularization parameter.
        beta: Coefficient vector where the logistic is evaluated.
    """
    logits = jnp.exp(X @ beta)
    logits = logits / (1 + logits)
    return (X * logits * (1 - logits)).T @ X + alpha * jnp.eye(beta.size)


def newton_logistic(X, y, alpha, beta0, max_iter=50, with_grad=False):
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

    hbeta = newton(grad_f, hess_f, beta0, max_iter=max_iter)

    if with_grad:
        jac_newton = jacfwd(newton_logistic, argnums=2)  # diff wrt alpha
        one_step_jac = jac_newton(X, y, alpha, hbeta, max_iter=1)
        return hbeta, one_step_jac
    else:
        return hbeta
