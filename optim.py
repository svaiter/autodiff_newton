import jax
import jax.numpy as jnp


def newton(grad_f, hessian_f, x0, lr=1.0, max_iter=50, callback=None):
    """Newton's method.
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


def WolfeLineSearch(f, g, d, x, c1, c2, min_step, max_step, max_iterations=50):
    """Wolfe line search.
    Args:
        f: function that computes the value of f.
        g: function that computes the gradient of f.
        d: search direction.
        x: current point.
        c1: parameter for Armijo condition.
        c2: parameter for curvature condition.
        min_step: minimum step size.
        max_step: maximum step size.
        max_iterations: maximum number of iterations.
    """
    # We start by precomputing some values
    step = (min_step + max_step) / 2
    y = f(x)
    m = jnp.sum(-d * d)

    # Define the first condition
    def check_condition_1(step):
        return f(x + step * d) > y + c1 * step * m

    # Define the second condition
    def check_condition_2(step):
        return jnp.sum(g(x + step * d) * d) < c2 * m

    # Then we divide the step size by 2 as long as one of the conditions is met
    condition_1 = check_condition_1(step)
    condition_2 = check_condition_2(step)
    n_iterations = 0

    while (condition_1 or condition_2) and n_iterations < max_iterations:

        if condition_1:
            max_step = step
        else:
            min_step = step

        step = (min_step + max_step) / 2

        condition_1 = check_condition_1(step)
        condition_2 = check_condition_2(step)
        n_iterations += 1

    return step
