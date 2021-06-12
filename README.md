# image-color-replacement
Image color replacement using numerical optimization.

## Usage
* `pip install -r requirements.txt`
* Edit `config.yml` (configuration options explanation below)
* `python image_color_replacement.py`

## Configuration
* `input_image_path`: Path of source (input) image
* `output_image_path`: Path to save target image
* `intermeidate_representation_path` (optional): Path to save intermediate image representations
* `from_colors`: List of RGB colors to replace
* `to_colors`: List of RGB colors with which to replace colors from `from_colors`, in matching order
* `fixed_colors`: List of RGB colors that shouldn't change
* `latent_color_components_k`: Number of latent color components
* `optimization_scale`: How much to downscale image before performing optimization (it is *not* a good idea to optimize on an unscaled large image)
* `wolfe_conditions`: Wolfe line search parameters (only used if BFGS implementation is "`scratch`")
* `penalty_weights`: List of penalty weights (weighting objective vs. constraint violation)
* `bfgs_max_iterations`: Max BFGS iterations (per penalty weight)
* `bfgs_implementation`: One of "`scipy`" and "`scratch`", "`scipy`" highly advised

## More
See [this post](https://andersource.dev/2021/06/12/image-color-replacement.html) for more details.
