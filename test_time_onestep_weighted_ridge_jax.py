import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from jax import jacfwd, value_and_grad
from jax.lax import stop_gradient
import jax.numpy as jnp

def generate_matrix(n, p, cond):
    X = np.random.randn(n, p)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S = np.diag(np.linspace(1, cond, min(n, p)))
    X = U @ S @ V
    return X

X = generate_matrix(100, 10, 10)
y = np.random.randn(100)
#X, y = fetch_libsvm("breast-cancer")
n, p = X.shape

big_step = 100

def gd(w0, theta, lr, alpha=1.0, start_trace=0, max_iter=100):
    w = w0
    dtheta = jnp.zeros((p,n))
    for i in range(start_trace):
        dw = X.T @ jnp.diag(stop_gradient(theta)) @ (X @ stop_gradient(w) - y) + alpha * stop_gradient(w)
        w = stop_gradient(w) - lr * stop_gradient(dw)
    for i in range(max_iter - start_trace):
        dw = X.T @ jnp.diag(theta) @ (X @ w - y) + alpha * w
        w = w - lr * dw
    return jnp.linalg.norm(w) ** 2
diff_gd = value_and_grad(gd, argnums=1)

str_traces = [0, 4*big_step, 5*big_step]
alpha_facs = [0.001, 0.01]
max_iter = 5*big_step + 1
theta = np.ones(n)
norm_dthetahats = np.ones((len(alpha_facs), len(str_traces), 2, 1+max_iter // big_step))
norm_ws = np.ones((len(alpha_facs), len(str_traces), 2, 1+max_iter // big_step))
times = np.zeros((len(alpha_facs), len(str_traces), 2, 1+max_iter // big_step))
for t, alpha_fac in enumerate(alpha_facs):
    alpha = (np.linalg.norm(X,2) ** 2) * alpha_fac
    eigs = np.sort(np.linalg.svd(X.T @ np.diag(theta) @ X + alpha * np.eye(p))[1])[::-1]
    L, mu = eigs[0], eigs[-1]
    true_eigs = np.sort(np.linalg.svd(X)[1])[::-1]
    true_cond = true_eigs[0]/true_eigs[-1]
    print(true_cond, L/mu)

    lrs = [1/L, 2/(mu+L)]
    n_repeats = 5

    wsklearn = np.linalg.solve(X.T @ np.diag(theta) @ X + alpha * np.eye(p), X.T @ np.diag(theta) @ y)
    sqwsk = np.linalg.norm(wsklearn) ** 2
    _, jac_gt = diff_gd(np.zeros(p), theta, 1/L, alpha=alpha, start_trace=0, max_iter=10000)

    for j, str_trace in enumerate(str_traces):
        for r, lr in enumerate(lrs):
            for ts in range(1+max_iter // big_step):
                t0 = time.time()
                what, dwhat = diff_gd(np.zeros(p), theta, lr, alpha=alpha, start_trace=str_trace, max_iter=ts * big_step)
                t1 = time.time()
                tid = t1 - t0
                norm_ws[t, j, r, ts] = np.linalg.norm(what - sqwsk) / np.linalg.norm(wsklearn)
                norm_dthetahats[t, j, r, ts] = np.linalg.norm(dwhat - jac_gt) / np.linalg.norm(jac_gt)
                times[t, j, r, ts] = tid

#### DISPLAY
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
lrs_s = ["$1/L$", "$2/(\mu + L)$"]
viridis = cm.get_cmap('viridis', len(str_traces))
colors = viridis(np.linspace(0, 1, len(str_traces)))
for t, alpha_fac in enumerate(alpha_facs):
    for r, lr in enumerate(lrs):
        for j, str_trace in enumerate(str_traces):
            ax[t, r].scatter(times[t,j,r,:].T, norm_dthetahats[t,j,r,:].T, color=colors[j])
            ax[t, r].plot(times[t,j,r,:].T, norm_dthetahats[t,j,r,:].T, '', label='_nolegend_', color=colors[j])
            ax[t, r].scatter(times[t,j,r,:].T, norm_ws[t,j,r,:].T, marker='x', label='_nolegend_', color=colors[j])
            ax[t, r].plot(times[t,j,r,:].T, norm_ws[t,j,r,:].T, '--', label='_nolegend_', color=colors[j])
        ax[t, r].set_yscale("log")
        ax[t, r].set_xlabel("Time (s)")
        ax[t, r].set_title(lrs_s[r])
        ax[t, 0].set_ylabel("Relative suboptimality")

legends = ["autodiff", "1-step", "implicit diff"]
fig.legend(legends, loc='upper right', bbox_to_anchor=(1,1), ncol=len(legends), bbox_transform=fig.transFigure)

plt.savefig("1step.pdf")
