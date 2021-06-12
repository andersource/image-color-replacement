import numpy as np
from numba import jit


@jit(nopython=True)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@jit(nopython=True)
def pack(T, B, N):
    """Flatten parameters to single 1D array"""
    return np.concatenate((T.ravel(), B.ravel(), N.ravel()))


@jit(nopython=True)
def unpack(X, K):
    """Extract parameters from single 1D array"""
    T = X[: 3 * K].reshape((3, K))
    B = X[3 * K : 6 * K].reshape((K, 3))
    N = X[6 * K :].reshape((K, 3))

    return T, B, N


@jit(nopython=True)
def calculate_results(I, from_colors, T, B, N):
    """
    Calculate forward pass.

    Return both complete transformations and intermediate representations.
    """
    I_A = sigmoid(I @ T)
    I_pred = sigmoid(I_A @ B)
    C2_A = sigmoid(from_colors @ T)
    C2_pred = sigmoid(C2_A @ N)
    return I_pred, C2_pred, I_A, C2_A


@jit(nopython=True)
def calculate_loss(I, to_colors, T, B, N, I_pred, C2_pred, I_A, C2_A, loss_weights):
    return (
        loss_weights[0] * np.abs(B).mean()
        + loss_weights[1] * np.abs(N).mean()
        + loss_weights[2] * np.abs(I_A).mean()
        + loss_weights[3] * np.sqrt(((I - I_pred) ** 2).sum(axis=1)).mean()
        + loss_weights[4] * np.sqrt(((to_colors - C2_pred) ** 2).sum(axis=1)).mean()
    )


@jit(nopython=True)
def collect_losses(partials, weights):
    assert len(weights) == len(partials)
    res = np.zeros(partials[0].shape)
    for i in range(len(weights)):
        res += weights[i] * partials[i]

    return res


# Gradient calculation (see derivations in presentation)
@jit(nopython=True)
def calculate_gradients(
    I, from_colors, to_colors, T, B, N, I_pred, C2_pred, I_A, C2_A, loss_weights
):
    I_diff_a = I_pred - I
    I_diff = (
        I_diff_a / np.sqrt((I_diff_a ** 2).sum(axis=1)).reshape((-1, 1)) / I.shape[0]
    )
    C2_diff_a = C2_pred - to_colors
    C2_diff = (
        C2_diff_a
        / np.sqrt((C2_diff_a ** 2).sum(axis=1)).reshape((-1, 1))
        / to_colors.shape[0]
    )

    dJdT = [
        np.zeros(T.shape),
        np.zeros(T.shape),
        I.T @ (np.sign(I_A) * I_A * (1 - I_A)) / I_A.size,
        dJdT_partial(T, I, I_diff, I_pred, B, I_A),
        dJdT_partial(T, from_colors, C2_diff, C2_pred, N, C2_A),
    ]

    dJdB = [
        np.sign(B) / B.size,
        np.zeros(B.shape),
        np.zeros(B.shape),
        dJdP_partial(I_diff, I_pred, I_A, B),
        np.zeros(B.shape),
    ]

    dJdN = [
        np.zeros(N.shape),
        np.sign(N) / N.size,
        np.zeros(N.shape),
        np.zeros(N.shape),
        dJdP_partial(C2_diff, C2_pred, C2_A, N),
    ]

    return (
        collect_losses(dJdT, loss_weights),
        collect_losses(dJdB, loss_weights),
        collect_losses(dJdN, loss_weights),
    )


@jit(nopython=True)
def dJdT_partial(T, X, diff, pred, P, A):
    res = np.zeros(T.shape)
    for i in range(T.shape[1]):
        for j in range(T.shape[0]):
            res[:, i] += diff[:, j] @ (
                (
                    P[i, j] * pred[:, j] * (1 - pred[:, j]) * A[:, i] * (1 - A[:, i])
                ).reshape((-1, 1))
                * X
            )

    return res


@jit(nopython=True)
def dJdP_partial(diff, pred, A, P):
    res = np.zeros(P.shape)
    for i in range(P.shape[1]):
        res[:, i] = diff[:, i] @ ((pred[:, i] * (1 - pred[:, i])).reshape((-1, 1)) * A)

    return res
