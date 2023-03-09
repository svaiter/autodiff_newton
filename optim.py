import jax
import jax.numpy as jnp


def newton(grad_f, hessian_f, x0, lr=1.0, max_iter=50, callback=None):
    """ Newton's method.
    Args:
        grad_f: function that computes the gradient of f.
        hessian_f: function that computes the Hessian of f.
        x0: initial guess.
        lr: learning rate.
        max_iter: maximum number of iterations.
        callback: callback function that is called after each iteration.
    """
    x = x0
    if callback:
        cx0 = callback(x0)
        callbacks = jnp.zeros((max_iter, *cx0.shape))
    for k in range(max_iter):
        g = grad_f(x)
        H = hessian_f(x)
        x = x - lr * jnp.linalg.solve(H, g)
        if callback:
            callbacks[k] = callback(x)
    if callback:
        return x, callbacks
    else:
        return x
