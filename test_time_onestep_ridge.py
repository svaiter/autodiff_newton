import time
import numpy as np
import matplotlib.pyplot as plt
from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge


X, y = fetch_libsvm("breast-cancer")
XTX = X.T @ X
XTXnorm = np.linalg.norm(XTX, 2)
XTy = X.T @ y
n, p = X.shape

def diff_gd(w0, theta, lr, start_trace=0, max_iter=100):
    t0 = time.time()
    times = np.zeros(max_iter // 100)
    w = w0
    dtheta = np.zeros(p)
    dthetas = np.zeros((max_iter // 100, p))
    for i in range(max_iter):
        if i % 100 == 0:
            dthetas[i//100, :] = dtheta
            times[i//100] = time.time() - t0
        dw = XTX @ w - XTy + theta * w
        w = w - lr * dw
        if i >= start_trace:
            dww = XTX + theta * np.eye(p)
            dwtheta = w
            dtheta = (np.eye(p) - lr * dww) @ dtheta - lr * dwtheta
    return w, dthetas, times

str_traces = [0, 1000, 2000, 3000, 4000, 5000]
theta_facs = [0.01, 0.001]
norm_dthetahats = np.ones((len(theta_facs), len(str_traces), n_repeats, max_iter // 100))
times = np.zeros((len(theta_facs), len(str_traces), n_repeats, max_iter // 100))
for t, theta_fac in enumerate(theta_facs):
    theta = (np.linalg.norm(X,2) ** 2) * theta_fac
    eigs = np.sort(np.linalg.svd(XTX + theta * np.eye(p))[1])[::-1]
    L, mu = eigs[0], eigs[-1]
    true_eigs = np.sort(np.linalg.svd(X)[1])[::-1]
    true_cond = true_eigs[0]/true_eigs[-1]

    lr = 2 / (mu + L)
    lrs = [1/L, np.sqrt(2)/L, 2/(mu+L)]
    max_iter = 5000
    n_repeats = 5

    wsklearn = np.linalg.solve(X.T @ X + theta * np.eye(p), X.T @ y)
    jac_gt = np.linalg.solve(XTX + theta * np.eye(p), -wsklearn)
    lossmin = ridge_loss_grads(wsklearn, theta)[0]

    t0 = time.time()
    _ = diff_gd(np.zeros(p), theta, lr, start_trace = max_iter, max_iter = max_iter)
    t1 = time.time()

    # norm_whats = np.ones((len(str_traces), n_repeats))
    for j, str_trace in enumerate(str_traces):
        for r, lr in enumerate(lrs):
            what, dthetahats, ltimes = diff_gd(np.zeros(p), theta, lr, start_trace=str_trace, max_iter=max_iter)
            norm_dthetahats[t, j, r, :] = np.linalg.norm(dthetahats - jac_gt, axis=1) / np.linalg.norm(jac_gt)
            times[t, j, r, :] = ltimes

#### DISPLAY

fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
lrs_s = ["$1/L$", "$\sqrt{2}/L$", "$2/(\mu + L)$"]
for t, theta_fac in enumerate(theta_facs):
    for r, lr in enumerate(lrs):
        for j, str_trace in enumerate(str_traces):
            ax[t, r].scatter(times[t,j,r,:].T, norm_dthetahats[t,j,r,:].T)
            ax[t, r].plot(times[t,j,r,:].T, norm_dthetahats[t,j,r,:].T, label='_nolegend_')
        ax[t, r].set_yscale("log")
        ax[t, r].set_xlabel("Time (s)")
        ax[t, r].set_title(lrs_s[r])
        ax[t, 0].set_ylabel("Relative Jacobian suboptimality")

legends = ["autodiff", "4-steps", "3-steps", "2-steps", "1-step", "no diff"]
fig.legend(legends, loc='upper right', bbox_to_anchor=(1,1), ncol=len(legends), bbox_transform=fig.transFigure)
  
plt.savefig("kstep.pdf")
