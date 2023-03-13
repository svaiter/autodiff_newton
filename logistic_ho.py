import jax.numpy as jnp
from jax import jacfwd
from optim import newton


def loss_logistic(X, y, alpha, beta):
    logits = jnp.exp(X @ beta)
    logits = logits / (1 + logits)
    return - jnp.sum(y * jnp.log(logits) + (1 - y) * jnp.log(1 - logits)) + alpha * jnp.sum(beta ** 2)


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
    return X.T @ jnp.diag(logits * (1-logits)) @ X + alpha * jnp.eye(beta.size)


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


def logistic_parameter_selection(X_train, X_val, y_train, y_val, rho, alpha0, max_iter=10):
    alpha = alpha0
    hbeta = jnp.zeros(X_train.shape[1])
    for i in range(max_iter):
        hbeta, hjac = newton_logistic(X_train, y_train, alpha, hbeta, with_grad=True)
        grad_outer = (grad_logistic(X_val, y_val, 0.0, hbeta).T @ hjac.T).T
        alpha = jnp.maximum(0, alpha - rho * grad_outer)
        loss = loss_logistic(X_val, y_val, alpha, hbeta)
        print(f"alpha:{alpha:.3f}\tloss:{loss:.3f}")
    return alpha
