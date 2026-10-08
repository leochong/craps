import numpy as np


def cma_es(f, x0, sigma0, lower, upper, popsize=None, maxiter=80, seed=0, tol=1e-9):
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = x0.size

    if popsize is None:
        popsize = 4 + int(3 * np.log(n))
    mu = popsize // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights ** 2)

    cc = 4.0 / (n + 4.0)
    cs = (mueff + 2.0) / (n + mueff + 3.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    invsqrtC = np.eye(n)

    def clip(x):
        return np.minimum(np.maximum(x, lower), upper)

    xmean = clip(x0.copy())
    sigma = sigma0
    best_x = xmean.copy()
    best_f = f(best_x)
    history = [{"iter": 0, "best_f": best_f, "sigma": sigma}]
    eigeneval = 0

    for it in range(1, maxiter + 1):
        arz = rng.standard_normal((popsize, n))
        arx = clip(xmean + sigma * (arz @ (B * D).T))
        fits = np.array([f(x) for x in arx])

        order = np.argsort(fits)
        if fits[order[0]] < best_f:
            best_f = float(fits[order[0]])
            best_x = arx[order[0]].copy()

        xold = xmean.copy()
        xmean = clip(np.sum(weights[:, None] * arx[order[:mu]], axis=0))
        y = (xmean - xold) / sigma

        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (invsqrtC @ y)
        hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * it)) / chiN
                < 1.4 + 2.0 / (n + 1.0))
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y

        artmp = (arx[order[:mu]] - xold) / sigma
        C = ((1.0 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
             + cmu * (artmp.T @ (weights[:, None] * artmp)))
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))

        if it - eigeneval > popsize / (c1 + cmu) / n / 10.0:
            eigeneval = it
            C = np.triu(C) + np.triu(C, 1).T
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-30))
            invsqrtC = B @ np.diag(1.0 / D) @ B.T

        history.append({"iter": it, "best_f": best_f, "sigma": float(sigma)})
        if sigma < tol:
            break

    return {"x": best_x, "f": best_f, "history": history}