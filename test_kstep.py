import numpy as np
import matplotlib.pyplot as plt
from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
import jax.numpy as jnp
from jax import jacfwd

def generate_matrix_with_cond(n, cond_P):
    # From https://gist.github.com/bstellato/23322fe5d87bb71da922fbc41d658079
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n + 1)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))[:n]
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)
    return P

X, y = fetch_libsvm("breast-cancer")
#X = X/np.linalg.norm(X, axis=0)
#X, y = make_regression(50, 100)
#X = generate_matrix_with_cond(100, 20.)
#y = np.random.randn(100)
XTX = X.T @ X
XTXnorm = np.linalg.norm(XTX, 2)
XTy = X.T @ y
n, p = X.shape

def ridge_loss_grads(w, theta):
    """Ridge loss"""
    loss = 0.5 * np.sum((X @ w - y) ** 2) + 0.5 * theta * np.sum(w * w)
    dw = XTX @ w - XTy + theta * w
    dww = XTX + theta * np.eye(p)
    dwtheta = w
    return loss, dw, dww, dwtheta


def logistic_loss_grads(w, theta):
    logits = np.exp(X @ w)
    logits = logits / (1 + logits)
    loss = -1 / n * np.sum(
        y * np.log(logits) + (1 - y) * np.log(1 - logits)
    ) + theta / 2 * np.sum(w**2)
    dw = 1 / n * X.T @ (logits - y) + theta * w
    dww = 1 / n * X.T @ np.diag(logits * (1 - logits)) @ X + theta * np.eye(p)
    dwtheta = w
    return loss, dw, dww, dwtheta


def diff_gd(w0, theta, lr, start_trace=0, max_iter=100):
    w = w0
    dtheta = np.zeros(p)
    ws = np.zeros((max_iter, p))
    dthetas = np.zeros((max_iter, p))
    sol = np.linalg.solve(X.T @ X + theta * np.eye(p), X.T @ y)
    for i in range(max_iter):
        loss, dw, dww, dwtheta = ridge_loss_grads(w, theta)
        ws[i,:] = w
        w = w - lr * dw
        if i >= start_trace:
            dtheta = (np.eye(p) - lr * dww) @ dtheta - lr * dwtheta
        dthetas[i,:] = dtheta
    return ws, dthetas

max_iter = 10000

theta = (np.linalg.norm(X,2) ** 2) * 0.001
eigs = np.sort(np.linalg.svd(XTX + theta * np.eye(p))[1])[::-1]
L, mu = eigs[0], eigs[-1]
true_eigs = np.sort(np.linalg.svd(X)[1])[::-1]
true_cond = true_eigs[0]/true_eigs[-1]

wsklearn = np.linalg.solve(X.T @ X + theta * np.eye(p), X.T @ y)
lossmin = ridge_loss_grads(wsklearn, theta)[0]
jac_gt = np.linalg.solve(XTX + theta * np.eye(p), -wsklearn)

lr = 2 / (mu + L)
str_traces = [0, 100, 150, 180, 200]#, 400, 600, 800, 995]
str_traces = [0, 1000, 2000, 3000, 4000]
norm_whats = np.ones((max_iter, len(str_traces)))
norm_dthetahats = np.ones((max_iter, len(str_traces)))
sub_whats = np.zeros(max_iter)
for j, str_trace in enumerate(str_traces):
    whats, dthetahats = diff_gd(np.zeros(p), theta, lr, start_trace=str_trace, max_iter=max_iter)
    norm_whats[:, j] = np.linalg.norm(whats - wsklearn , axis=1) / np.linalg.norm(wsklearn)
    norm_dthetahats[:, j] = np.linalg.norm(dthetahats - jac_gt, axis=1) / np.linalg.norm(jac_gt)

max_display = 5000
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].semilogy(norm_whats[:max_display])
ax[1].semilogy(norm_dthetahats[:max_display])
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Relative iterates suboptimality")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Relative Jacobian suboptimality")
ax[0].set_title(f"$\\kappa = {L/mu:.2f}, \\kappa_T = {true_cond:.2f}$")
    
plt.show()
