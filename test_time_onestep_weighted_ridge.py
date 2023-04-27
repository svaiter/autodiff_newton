import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression

def generate_matrix(n, p, cond):
    X = np.random.randn(n, p)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S = np.diag(np.linspace(1, cond, min(n, p)))
    X = U @ S @ V
    return X

X,y = make_regression(400,60,effective_rank=2)
#X,y = make_regression(2000,200,effective_rank=2)
# y = np.random.randn(100)
X, y = fetch_libsvm("cpusmall")
n, p = X.shape

big_step = 1000

def diff_gd(w0, theta, lr, alpha=1.0, start_trace=0, max_iter=100):
    t0 = time.time()
    times = np.zeros(1+max_iter // big_step)
    w = w0
    ws = np.zeros((1+max_iter // big_step, p))
    dtheta = np.zeros((p,n))
    dthetas = np.zeros((1+max_iter // big_step, p, n))
    for i in range(max_iter):
        if i % big_step == 0:
            dthetas[i//big_step, :, :] = dtheta
            ws[i//big_step, :] = w
            times[i//big_step] = time.time() - t0
        dw = X.T @ np.diag(theta) @ (X @ w - y) + alpha * w
        w = w - lr * dw
        if i >= start_trace:
            dww = X.T @ np.diag(theta) @ X + alpha * np.eye(p)
            dwtheta = np.diag(w) @ (X.T * X.T)
            dtheta = (np.eye(p) - lr * dww) @ dtheta - lr * dwtheta
    return w, ws, dthetas, times

str_traces = [0, 4*big_step, 5*big_step]
alpha_facs = [1e-3, 1e-2]
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
    wsklearn = np.linalg.solve(X.T @ np.diag(theta) @ X + alpha * np.eye(p), X.T @ np.diag(theta) @ y)
    jac_gt = np.linalg.solve(X.T @ np.diag(theta) @ X + alpha * np.eye(p), -np.diag(wsklearn) @ (X.T * X.T))

    for j, str_trace in enumerate(str_traces):
        for r, lr in enumerate(lrs):
            what, ws, dthetahats, ltimes = diff_gd(np.zeros(p), theta, lr, alpha=alpha, start_trace=str_trace, max_iter=max_iter)
            norm_ws[t, j, r, :] = np.linalg.norm(ws - wsklearn, axis=1) / np.linalg.norm(wsklearn)
            if str_trace == max_iter-1:
                tid0 = time.time()
                did = np.linalg.solve(X.T @ np.diag(theta) @ X + alpha * np.eye(p), -np.diag(what) @ (X.T * X.T))
                dthetahats[-1,:,:] = did
                tid = time.time() - tid0
            else:
                tid = 0
            norm_dthetahats[t, j, r, :] = np.linalg.norm(dthetahats - jac_gt, axis=(1,2)) / np.linalg.norm(jac_gt)
            times[t, j, r, :] = ltimes + tid

#### DISPLAY
import matplotlib
matplotlib.rcParams.update({'font.size': 10, 'figure.dpi':300})
fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
lrs_s = ["$1/L$", "$2/(\mu + L)$"]
legends = ["autodiff (Jacobian)", "autodiff (iter.)", "1-step (Jacobian)", "1-step (iter.)", "implicit diff (Jacobian)", "implicit diff (iter.)"]
viridis = cm.get_cmap('viridis', len(str_traces))
colors = viridis(np.linspace(0, 1, len(str_traces)))
for t, alpha_fac in enumerate(alpha_facs):
    for r, lr in enumerate(lrs):
        for j, str_trace in enumerate(str_traces):
            axs[t, r].scatter(times[t,j,r,:].T, norm_dthetahats[t,j,r,:].T, color=colors[j])
            axs[t, r].plot(times[t,j,r,:].T, norm_dthetahats[t,j,r,:].T, label='_nolegend_', color=colors[j])
            axs[t, r].scatter(times[t,j,r,:].T, norm_ws[t,j,r,:].T, marker='x', color=colors[j])
            axs[t, r].plot(times[t,j,r,:].T, norm_ws[t,j,r,:].T, '--', label='_nolegend_', color=colors[j])
        axs[t, r].set_yscale("log")
        axs[t, r].set_xlabel("Time (s)")
        axs[0, r].set_title(lrs_s[r])
        axs[t, 0].set_ylabel("Relative suboptimality")

fig.legend(legends, loc='upper right', bbox_to_anchor=(1,1), ncol=len(legends), bbox_transform=fig.transFigure)

plt.savefig("1step.pdf",bbox_inches='tight')
