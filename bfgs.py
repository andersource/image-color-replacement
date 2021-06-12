from tqdm import tqdm
import numpy as np
from numba import jit
from problem_math import (
    pack,
    unpack,
    calculate_results,
    calculate_loss,
    calculate_gradients,
)


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
    new_forward_res = None
    new_dJdT = new_dJdB = new_dJdN = None
    losses = []
    H = np.eye(9 * K)  # Initial Hessian approximation (identity matrix)
    for iter_idx in tqdm(
        range(max_iterations), desc=f"Running BFGS for penalty {penalty_weight}"
    ):
        curr_X = pack(T, B, N)
        if (
            new_forward_res is None
        ):  # Need to calculate the "forward pass" here only on the first time,
            # in consecutive iterations we will have calculated this in the previous iteration
            # for the Hessian approx. update
            forward_res = calculate_results(I, from_colors, T, B, N)
        else:
            forward_res = new_forward_res

        curr_loss = calculate_loss(I, to_colors, T, B, N, *forward_res, loss_weights)
        losses.append(curr_loss)
        if (
            new_dJdT is None
        ):  # Again we only need to calculate the gradients here on the first iteration
            dJdT, dJdB, dJdN = calculate_gradients(
                I, from_colors, to_colors, T, B, N, *forward_res, loss_weights
            )
        else:
            dJdT, dJdB, dJdN = new_dJdT, new_dJdB, new_dJdN

        dJdX = pack(dJdT, dJdB, dJdN)
        p = -np.linalg.inv(H) @ dJdX  # Find movement direction

        chosen_alpha = line_search(
            I,
            from_colors,
            to_colors,
            loss_weights,
            alphas,
            curr_X,
            curr_loss,
            dJdX,
            K,
            p,
            wolfe_c1,
            wolfe_c2,
        )
        if (
            chosen_alpha is None
        ):  # No alpha candidates which hold the Wolfe conditions - stop optimizing
            break

        s = chosen_alpha * p
        new_X = curr_X + s
        T, B, N = unpack(
            new_X, K
        )  # Update parameters according to the chosen direction and step size
        new_forward_res = calculate_results(I, from_colors, T, B, N)

        # Hessian approx. update
        new_dJdT, new_dJdB, new_dJdN = calculate_gradients(
            I, from_colors, to_colors, T, B, N, *new_forward_res, loss_weights
        )
        new_dJdX = pack(new_dJdT, new_dJdB, new_dJdN)
        y = new_dJdX - dJdX
        s = s.reshape((-1, 1))
        y = y.reshape((-1, 1))
        H = H - (H @ s @ s.T @ H) / (s.T @ H @ s) + (y @ y.T) / (y.T @ s)

    return T, B, N, losses


@jit(nopython=True)
def line_search(
    I,
    from_colors,
    to_colors,
    loss_weights,
    alphas,
    curr_X,
    curr_loss,
    dJdX,
    K,
    p,
    wolfe_c1,
    wolfe_c2,
):
    alpha_losses = np.ones(alphas.shape[0]) * np.inf
    for i, alpha in enumerate(alphas):
        # For each candidate alpha, calculate loss at that alpha and check the Wolfe conditions
        temp_X = curr_X + alpha * p
        temp_T, temp_B, temp_N = unpack(temp_X, K)
        temp_forward_res = calculate_results(I, from_colors, temp_T, temp_B, temp_N)
        temp_dJdT, temp_dJdB, temp_dJdN = calculate_gradients(
            I,
            from_colors,
            to_colors,
            temp_T,
            temp_B,
            temp_N,
            *temp_forward_res,
            loss_weights,
        )
        temp_dJdX = pack(temp_dJdT, temp_dJdB, temp_dJdN)
        temp_loss = calculate_loss(
            I, to_colors, temp_T, temp_B, temp_N, *temp_forward_res, loss_weights
        )
        if temp_loss <= curr_loss + wolfe_c1 * alpha * (
            dJdX @ p
        ) and temp_dJdX @ p >= wolfe_c2 * (dJdX @ p):
            alpha_losses[i] = temp_loss

    if np.isinf(
        np.min(alpha_losses)
    ):  # No alpha candidates for which the Wolfe conditions hold
        return None

    return alphas[np.argmin(alpha_losses)]
