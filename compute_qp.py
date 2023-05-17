# from matplotlib import pyplot as plt
# plt.rcParams['figure.figsize'] = [12, 8]
import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp

import time
import timeit
import gc

from jax.config import config

config.update("jax_enable_x64", True)

from QPIPutils import *


repeat = 5  # 10
repMean = 2  # 5
nExp = 4


nMin = 100
pMin = 50
mMin = 25


nMax = 1000
pMax = 500
mMax = 250

ns = np.int32(
    np.round(
        np.exp(
            np.arange(nExp + 1) / nExp * (np.log(nMax) - np.log(nMin)) + np.log(nMin)
        )
    )
)
ms = np.int32(
    np.round(
        np.exp(
            np.arange(nExp + 1) / nExp * (np.log(mMax) - np.log(mMin)) + np.log(mMin)
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
    n = ns[exp_id]
    m = ms[exp_id]
    p = ps[exp_id]
    nParam = n * n + n * p + n * m + n + m + p

    forwardTime = np.zeros(repeat)
    diffImpTime = np.zeros(repeat)
    diffOneStepTime = np.zeros(repeat)
    diffAutodiffTime = np.zeros(repeat)
    print(exp_id, end="")
    for rep in range(repeat):
        print(".", end="")
        Q, A, G, q, b, h = generateRandomQPdata(n, m, p)

        zou = gc.collect()
        x, s, z, y, _, _ = IP(Q, A, G, q, b, h, stopGrad=True)
        forwardTime[rep] = np.mean(
            timeit.repeat(
                "x, s, z, y, _, _ = IP(Q, A, G, q, b, h, stopGrad= True)",
                globals=globals(),
                number=1,
                repeat=repMean,
            )
        )

        zou = gc.collect()
        zou = diffImpNorm(Q, A, G, q, b, h)
        diffImpTime[rep] = np.mean(
            timeit.repeat(
                "diffImpNorm(Q, A, G, q, b, h)",
                globals=globals(),
                number=1,
                repeat=repMean,
            )
        )

        zou = gc.collect()
        zou = diffIPNorm(Q, A, G, q, b, h)
        diffOneStepTime[rep] = np.mean(
            timeit.repeat(
                "diffIPNorm(Q, A, G, q, b, h)",
                globals=globals(),
                number=1,
                repeat=repMean,
            )
        )

        zou = gc.collect()
        zou = diffIPNorm(Q, A, G, q, b, h, stopGrad=False)
        diffAutodiffTime[rep] = np.mean(
            timeit.repeat(
                "diffIPNorm(Q, A, G, q, b, h, stopGrad = False)",
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
        diffImpNorm(Q, A, G, q, b, h) - diffIPNorm(Q, A, G, q, b, h)
    )
    res[exp_id, 10] = np.linalg.norm(
        diffImpNorm(Q, A, G, q, b, h) - diffIPNorm(Q, A, G, q, b, h, stopGrad=False)
    )
    np.savetxt("resDiffIP.csv", res, delimiter=",")
