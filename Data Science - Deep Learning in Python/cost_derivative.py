from numpy import log, zeros, outer

def cost(T, Y):
    total = T * log(Y)
    return(tot.sum())

def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]
    # Slow
    #ret1 = zeros((M, K))
    #for n in range(N):
    #    for m in range(M):
    #        for k in range(K):
    #            ret1[m, k] += (T[n, k] - Y[n, k]) * Z[n, m]

    # ret2 = zeros((M, K))
    # for n in range(N):
    #     for k in range(K):
    #         ret2[:, k] += (T[n, k] - Y[n, k]) * Z[n, :]

    # ret3 = zeros((M, K))
    # for n in range(N):
    #     ret3 += outer(Z[n], T[n] - Y[n])

    ret4 = Z.T.dot(T - Y)

    return ret4

def derivative_b2(T, Y):
    return (T-Y).sum(axis = 0)

def derivative_w1(X, Z, T, Y, W2):
    # N, D = X.shape
    # M, K = W2.shape
    # ret1 = zeros((D, M))
    # for n in range(N):
    #     for k in range(K):
    #         for m in range(M):
    #             for d in range(D):
    #                 ret1[d, m] += (T[n, k] - Y[n, k]) * W2[m, k] * Z[n, m]*(1 - Z[n, m]) * X[n, d]

    dz = (T - Y).dot(W2.T) * Z * (1-Z)
    ret1 = X.T.dot(dz)

    return ret1

def derivative_b1(T, Y, W2, Z):
    return ((T-Y).dot(W2.T) * Z * (1-Z)).sum(axis = 0)
