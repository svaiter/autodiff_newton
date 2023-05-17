import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp
import sklearn.preprocessing

import time
import timeit
import gc

from jax.config import config

config.update("jax_enable_x64", True)

from NewtonLogisticutils import *


repeat = 5  # 10
repMean = 2  # 5
nExp = 4

pMin = 50
n1Min = 50
n2Min = 50

pMax = 1000
n1Max = 1000
n2Max = 1000

n1s = np.int32(
    np.round(
        np.exp(
            np.arange(nExp + 1) / nExp * (np.log(n1Max) - np.log(n1Min)) + np.log(n1Min)
        )
    )
)
n2s = np.int32(
    np.round(
        np.exp(
            np.arange(nExp + 1) / nExp * (np.log(n2Max) - np.log(n2Min)) + np.log(n2Min)
        )
    )
)
ps = np.int32(
    np.round(
        np.exp(
            np.arange(nExp + 1) / nExp * (np.log(pMax) - np.log(pMin)) + np.log(pMin)
        )
    )
)


res = np.zeros((nExp + 1, 11))

for exp_id in range(nExp + 1):
    n1 = n1s[exp_id]
    n2 = n2s[exp_id]
    n = n1 + n2
    pDim = ps[exp_id]
    nParam = pDim * (n1 + n2) + (n1 + n2)

    forwardTime = np.zeros(repeat)
    diffImpTime = np.zeros(repeat)
    diffOneStepTime = np.zeros(repeat)
    diffAutodiffTime = np.zeros(repeat)
    print(exp_id, end="")
    for rep in range(repeat):
        print(".", end="")
        lbda = jnp.array(-5.0)
        w = jnp.array(np.ones((n, 1)) * 1 / n)
        X, y = generateData(n1, n2, pDim - 1)
        betaInit = np.zeros((pDim, 1))
        betaInit[0] = 0.5
        betaZero = jnp.array(betaInit)
        lbda = jnp.array(-5.0)

        zou = gc.collect()
        beta = newton(X, y, betaZero, lbda, w, stopGrad=True)
        forwardTime[rep] = np.mean(
            timeit.repeat(
                "newton(X,y,betaZero,lbda, w, stopGrad = True)",
                globals=globals(),
                number=1,
                repeat=repMean,
            )
        )

        zou = gc.collect()
        s1 = diffImpNorm(X, y, betaZero, lbda, w)
        diffImpTime[rep] = np.mean(
            timeit.repeat(
                "diffImpNorm(X,y,betaZero,lbda, w)",
                globals=globals(),
                number=1,
                repeat=repMean,
            )
        )

        zou = gc.collect()
        zou = diffNewtonNorm(X, y, betaZero, lbda, w)
        diffOneStepTime[rep] = np.mean(
            timeit.repeat(
                "diffNewtonNorm(X,y,betaZero,lbda, w)",
                globals=globals(),
                number=1,
                repeat=repMean,
            )
        )

        zou = gc.collect()
        zou = diffNewtonNorm(X, y, betaZero, lbda, w, stopGrad=False)
        diffAutodiffTime[rep] = np.mean(
            timeit.repeat(
                "diffNewtonNorm(X,y,betaZero,lbda, w, stopGrad = False)",
                globals=globals(),
                number=1,
                repeat=repMean,
            )
        )
    print(".")
    res[exp_id, 0] = nParam
    res[exp_id, 1] = np.mean(forwardTime)
    res[exp_id, 2] = np.std(forwardTime)
    res[exp_id, 3] = np.mean(diffImpTime)
    res[exp_id, 4] = np.std(diffImpTime)
    res[exp_id, 5] = np.mean(diffOneStepTime)
    res[exp_id, 6] = np.std(diffOneStepTime)
    res[exp_id, 7] = np.mean(diffAutodiffTime)
    res[exp_id, 8] = np.std(diffAutodiffTime)
    res[exp_id, 9] = np.linalg.norm(
        diffImpNorm(X, y, betaZero, lbda, w) - diffNewtonNorm(X, y, betaZero, lbda, w)
    )
    res[exp_id, 10] = np.linalg.norm(
        diffImpNorm(X, y, betaZero, lbda, w)
        - diffNewtonNorm(X, y, betaZero, lbda, w, stopGrad=False)
    )
    np.savetxt("resDiffNewton.csv", res, delimiter=",")
