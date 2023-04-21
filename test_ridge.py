import numpy as np
import matplotlib.pyplot as plt
from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
import jax.numpy as jnp
from jax import jacfwd

X, y = fetch_libsvm("breast-cancer")
#X = X/np.linalg.norm(X, axis=0)
X, y = make_regression(200, 100)
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


def diff_gd(w0, theta, lr, max_iter=100):
    w = w0
    dtheta = np.zeros(p)
    ws = np.zeros((max_iter, p))
    dthetas = np.zeros((max_iter, p))
    losses = np.zeros(max_iter)
    sol = np.linalg.solve(X.T @ X + theta * np.eye(p), X.T @ y)
    for i in range(max_iter):
        loss, dw, dww, dwtheta = ridge_loss_grads(w, theta)
        ws[i,:] = w
        w = w - lr * dw
        losses[i] = loss
        # losses[i] = np.linalg.norm((w -sol).T @ (XTX + theta * np.eye(p)) @ (w -sol) )
        dtheta = (np.eye(p) - lr * dww) @ dtheta - lr * dwtheta
        dthetas[i,:] = dtheta
    return ws, dthetas, losses

# to check the derivative
def jax_gd(w0, theta, lr, max_iter=100):
    w = w0
    ws = []
    for i in range(max_iter):
        w = w - lr * (XTX @ w - XTy + theta * w)
        ws.append(w)
    return jnp.array(ws)
jac_jax_gd = jacfwd(jax_gd, argnums=1)

max_iter = 10000

theta = (np.linalg.norm(X,2) ** 2) * 0.1
eigs = np.sort(np.linalg.svd(XTX + theta * np.eye(p))[1])[::-1]
L, mu = eigs[0], eigs[-1]
clf = Ridge(alpha = theta, fit_intercept=False, tol=1e-12, max_iter=10000)
clf.fit(X, y)
wsklearn = clf.coef_
# wsklearn = np.linalg.solve(X.T @ X + theta * np.eye(p), X.T @ y)
lossmin = ridge_loss_grads(wsklearn, theta)[0]
jac_gt = np.linalg.solve(XTX + theta * np.eye(p), -wsklearn)

lrs = [1/L, np.sqrt(2)/L, 2. / (mu + L)]
lrs = np.linspace(1/L, 2. / (mu + L), num=10)
norm_whats = np.zeros((max_iter, len(lrs)))
norm_dthetahats = np.ones((max_iter, len(lrs)))
sub_whats = np.zeros((max_iter, len(lrs)))
for j, lr in enumerate(lrs):
    whats, dthetahats, losses = diff_gd(np.zeros(p), theta, lr, max_iter=max_iter)
    norm_whats[:, j] = np.linalg.norm(whats - wsklearn, axis=1) / np.linalg.norm(wsklearn)
    norm_dthetahats[:, j] = np.linalg.norm(dthetahats - jac_gt, axis=1) / np.linalg.norm(jac_gt)
    sub_whats[:, j] = losses / np.linalg.norm((-wsklearn).T @ (XTX + theta * np.eye(p)) @ (-wsklearn) )# - lossmin

max_display = 200
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
for j, lr in enumerate(lrs):
    color = plt.cm.viridis(j/len(lrs))
    ax[0].semilogy(norm_whats[:max_display, j], color=color)
    ax[1].semilogy(norm_dthetahats[:max_display, j], color=color)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Iterates suboptimality")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Jacobian suboptimality")
    
plt.savefig("4edouard.png")
plt.show()
