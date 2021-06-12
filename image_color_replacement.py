import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from yaml import load as load_yaml, CLoader
import warnings
from problem_math import sigmoid
import os


warnings.filterwarnings("ignore")  # Silence Numba performance warnings


def main():
    with open("config.yml", "r") as f:
        config = load_yaml(f, Loader=CLoader)

    if config["bfgs_implementation"] == "scipy":
        from bfgs_scipy import run_bfgs
    elif config["bfgs_implementation"] == "scratch":
        from bfgs import run_bfgs
    else:
        exit(
            "Unknown 'bfgs_implementation' specified, must be one of 'scipy', 'scratch'."
        )

    original_img = Image.open(config["input_image_path"])
    img = original_img.resize(
        tuple(
            (np.array(original_img.size) * config["optimization_scale"])
            .astype(int)
            .tolist()
        )
    )
    img = np.array(img) / 255

    pre_from_colors = np.array(config["from_colors"]) / 255
    to_colors = np.array(config["to_colors"]) / 255
    pre_fixed_colors = np.array(config["fixed_colors"]) / 255

    I = img.reshape((-1, 3))

    # Find colors in image which are closest to user-provided fixed colors
    fixed_colors_indices = [
        np.linalg.norm(I - pre_fixed_colors[i], axis=1).argmin()
        for i in range(pre_fixed_colors.shape[0])
    ]

    fixed_colors = I[fixed_colors_indices]

    # Find colors in image which are closest to user-provided "from" colors
    from_color_indices = [
        np.linalg.norm(I - pre_from_colors[i], axis=1).argmin()
        for i in range(pre_from_colors.shape[0])
    ]
    from_colors = I[from_color_indices]

    total_from_colors = np.concatenate([from_colors, fixed_colors])
    total_to_colors = np.concatenate([to_colors, fixed_colors])

    K = config["latent_color_components_k"]

    T = np.random.normal(size=(3, K))
    B = np.random.normal(size=(K, 3))
    N = np.random.normal(size=(K, 3))

    ALPHAS = np.exp(np.linspace(-12, 2, 20))

    WOLFE_C1 = config["wolfe_conditions"]["c1"]
    WOLFE_C2 = config["wolfe_conditions"]["c2"]

    complete_losses = []

    for curr_penalty_weight in config["penalty_weights"]:
        T, B, N, losses = run_bfgs(
            I,
            T,
            B,
            N,
            K,
            total_from_colors,
            total_to_colors,
            config["bfgs_max_iterations"],
            WOLFE_C1,
            WOLFE_C2,
            ALPHAS,
            curr_penalty_weight,
        )
        if losses is not None:
            complete_losses += losses

    I2_input = (np.array(original_img) / 255).reshape((-1, 3))

    intermediate_representation = sigmoid(I2_input @ T)
    original_img_shape = original_img.size[::-1]
    intermediate_representation_path = config.get(
        "intermeidate_representation_path", None
    )
    if intermediate_representation_path is not None:
        for channel in range(K):
            img = Image.fromarray(
                (
                    intermediate_representation[..., channel].reshape(
                        original_img_shape
                    )
                    * 255
                ).astype(np.uint8)
            )
            img.save(
                os.path.join(intermediate_representation_path, f"channel{channel}.png")
            )

    I2 = sigmoid(intermediate_representation @ N)
    converted_img = I2.reshape(np.array(original_img).shape)

    Image.fromarray((converted_img * 255).astype(np.uint8)).save(
        config["output_image_path"]
    )

    if len(complete_losses) > 2:
        plt.plot(complete_losses)
        plt.title("Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()


if __name__ == "__main__":
    main()
