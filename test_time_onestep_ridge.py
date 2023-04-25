import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

def generate_matrix(n, p, cond):
    X = np.random.randn(n, p)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S = np.diag(np.linspace(1, cond, min(n, p)))
    X = U @ S @ V
    return X

#X, y = fetch_libsvm("breast-cancer")
X = generate_matrix(200, 100, 1000.0)
y = np.random.randn(200)
XTX = X.T @ X
XTXnorm = np.linalg.norm(XTX, 2)
XTy = X.T @ y
n, p = X.shape

def diff_gd(w0, theta, lr, start_trace=0, max_iter=100):
    t0 = time.time()
    times = np.zeros(1+max_iter // 1000)
    w = w0
    ws = np.zeros((1+max_iter // 1000, p))
    dtheta = np.zeros(p)
    dthetas = np.zeros((1+max_iter // 1000, p))
    for i in range(max_iter):
        if i % 1000 == 0:
            dthetas[i//1000, :] = dtheta
            ws[i//1000, :] = w
            times[i//1000] = time.time() - t0
        dw = XTX @ w - XTy + theta * w
        w = w - lr * dw
        if i >= start_trace:
            dww = XTX + theta * np.eye(p)
            dwtheta = w
            dtheta = (np.eye(p) - lr * dww) @ dtheta - lr * dwtheta
    return w, ws, dthetas, times

str_traces = [0, 4000, 5000]
theta_facs = [0.01, 0.001]
theta_facs = [0.001, 1.0]
max_iter = 5001
norm_dthetahats = np.ones((len(theta_facs), len(str_traces), 2, 1+max_iter // 1000))
norm_ws = np.ones((len(theta_facs), len(str_traces), 2, 1+max_iter // 1000))
times = np.zeros((len(theta_facs), len(str_traces), 2, 1+max_iter // 1000))
for t, theta_fac in enumerate(theta_facs):
    theta = (np.linalg.norm(X,2) ** 2) * theta_fac
    eigs = np.sort(np.linalg.svd(XTX + theta * np.eye(p))[1])[::-1]
    L, mu = eigs[0], eigs[-1]
    true_eigs = np.sort(np.linalg.svd(X)[1])[::-1]
    true_cond = true_eigs[0]/true_eigs[-1]
    print(true_cond, L/mu)

    lr = 2 / (mu + L)
    lrs = [1/L, 2/(mu+L)]
    n_repeats = 5

    wsklearn = np.linalg.solve(X.T @ X + theta * np.eye(p), X.T @ y)
    jac_gt = np.linalg.solve(XTX + theta * np.eye(p), -wsklearn)

    # t0 = time.time()
    # _ = diff_gd(np.zeros(p), theta, lr, start_trace = max_iter, max_iter = max_iter)
    # t1 = time.time()

    for j, str_trace in enumerate(str_traces):
        for r, lr in enumerate(lrs):
            what, ws, dthetahats, ltimes = diff_gd(np.zeros(p), theta, lr, start_trace=str_trace, max_iter=max_iter)
            norm_ws[t, j, r, :] = np.linalg.norm(ws - wsklearn, axis=1) / np.linalg.norm(wsklearn)
            if str_trace == max_iter-1:
                tid0 = time.time()
                did = np.linalg.solve(XTX + theta * np.eye(p), -what)
                dthetahats[-1,:] = did
                tid = time.time() - tid0
            else:
                tid = 0
            norm_dthetahats[t, j, r, :] = np.linalg.norm(dthetahats - jac_gt, axis=1) / np.linalg.norm(jac_gt)
            times[t, j, r, :] = ltimes + tid

#### DISPLAY

fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
lrs_s = ["$1/L$", "$2/(\mu + L)$"]
viridis = cm.get_cmap('viridis', len(str_traces))
colors = viridis(np.linspace(0, 1, len(str_traces)))
for t, theta_fac in enumerate(theta_facs):
    for r, lr in enumerate(lrs):
        for j, str_trace in enumerate(str_traces):
            ax[t, r].scatter(times[t,j,r,:].T, norm_dthetahats[t,j,r,:].T, color=colors[j])
            ax[t, r].plot(times[t,j,r,:].T, norm_dthetahats[t,j,r,:].T, '', label='_nolegend_', color=colors[j])
            ax[t, r].scatter(times[t,j,r,:].T, norm_ws[t,j,r,:].T, marker='x', label='_nolegend_', color=colors[j])
            ax[t, r].plot(times[t,j,r,:].T, norm_ws[t,j,r,:].T, '--', label='_nolegend_', color=colors[j])
            # ax[t, r].scatter(times[t,j,r,:].T, np.exp(0.5*(np.log(norm_dthetahats[t,j,r,:].T) + np.log(norm_ws[t,j,r,:].T))), color=colors[j])
            # ax[t, r].plot(times[t,j,r,:].T, np.exp(0.5*(np.log(norm_dthetahats[t,j,r,:].T) + np.log(norm_ws[t,j,r,:].T))), label='_nolegend_', color=colors[j])
        ax[t, r].set_yscale("log")
        ax[t, r].set_xlabel("Time (s)")
        ax[t, r].set_title(lrs_s[r])
        ax[t, 0].set_ylabel("Relative Jacobian suboptimality")

legends = ["autodiff", "1-step", "implicit diff"]
fig.legend(legends, loc='upper right', bbox_to_anchor=(1,1), ncol=len(legends), bbox_transform=fig.transFigure)

plt.savefig("1step.pdf")
