from tqdm import tqdm
import numpy as np
from problem_math import (
    pack,
    unpack,
    calculate_results,
    calculate_loss,
    calculate_gradients,
)
from scipy.optimize import minimize


def run_bfgs(
    I,
    T,
    B,
    N,
    K,
    from_colors,
    to_colors,
    max_iterations,
    wolfe_c1,
    wolfe_c2,
    alphas,
    penalty_weight,
):
    loss_weights = np.array(
        [1, 1, 1, penalty_weight, penalty_weight]
    )  # Relative weight of each loss component

    def f_and_jac(x):
        t, b, n = unpack(x, K)
        forward_res = calculate_results(I, from_colors, t, b, n)
        dJdt, dJdb, dJdn = calculate_gradients(
            I, from_colors, to_colors, t, b, n, *forward_res, loss_weights
        )
        return calculate_loss(I, to_colors, t, b, n, *forward_res, loss_weights), pack(
            dJdt, dJdb, dJdn
        )

    opt_res = minimize(
        f_and_jac, pack(T, B, N), jac=True, options={"maxiter": max_iterations}
    )
    T, B, N = unpack(opt_res.x, K)

    return T, B, N, None
