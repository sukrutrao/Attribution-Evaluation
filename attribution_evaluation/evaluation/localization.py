import torch
from . import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


def get_localization_score_single(positive_attributions, head_idx, head_pos_idx, scale=2):
    """
    Evaluates the localization score for a set of positive attributions at a given classification head

    :param positive_attributions: Positive attributions
    :type positive_attributions: torch.Tensor of the shape (B, K, 1, H, W), where B is the batch size, K is the number of grid cells per image, H is the image height, and W is the image width
    :param head_idx: Index of the classification head in the attribution tensor
    :type head_idx: int, in the range [0, K-1]
    :param head_pos_idx: Index of the position of the grid cell in the grid
    :type head_pos_idx: int, in the range [0, N-1], for a N x N grid
    :param scale: Grid dimension N, defaults to 2
    :type scale: int, optional
    :return: Localization scores and sum of attributions inside each target grid cell
    :rtype: tuple consisting of two torch.Tensor each of the shape (B,)
    """
    original_img_dims = positive_attributions.shape[3] // scale, positive_attributions.shape[4] // scale
    row_idx = head_pos_idx // scale
    col_idx = head_pos_idx % scale
    positive_attributions_inside = positive_attributions[:, head_idx, :, row_idx * original_img_dims[0]:(
        row_idx + 1) * original_img_dims[0], col_idx * original_img_dims[1]:(col_idx + 1) * original_img_dims[1]].sum(dim=(1, 2, 3))
    positive_attributions_total = positive_attributions[:, head_idx].sum(
        dim=(1, 2, 3))
    return torch.where(positive_attributions_total > 1e-7, positive_attributions_inside / positive_attributions_total, torch.tensor(0.0)), positive_attributions_inside


def get_localization_score(attributions, only_corners=False, img_dims=(224, 224), scale=2):
    """
    Evaluates the localization score from a set of attributions

    :param attributions: Attributions
    :type attributions: torch.Tensor of the shape (B, K, 1, H, W), where B is the batch size, K is the number of grid cells per image, H is the image height, and W is the image width
    :param only_corners: Flag to enable evaluating localization only on the top-left and bottom-right corners of the grid, defaults to False
    :type only_corners: bool, optional
    :param img_dims: Dimensions of each grid cell, defaults to (224, 224)
    :type img_dims: tuple, optional. Dimensions (h, w), where h is the height of the grid cell, and w is the width of the grid cell.
    :param scale: Grid dimension N, defaults to 2
    :type scale: int, optional
    :return: Localization scores and sum of attributions inside each target grid cell
    :rtype: tuple consisting of two torch.Tensor each of the shape (B*K,)
    """
    grid_img_dims = tuple([scale * dim for dim in img_dims])
    interpolated_attributions = utils.interpolate_attributions(
        attributions, img_dims=grid_img_dims)
    positive_attributions = utils.get_positive_attributions(
        interpolated_attributions)
    grid_size = scale * scale
    if only_corners:
        head_list = [0, grid_size - 1]
    else:
        head_list = np.arange(grid_size).tolist()
    localization_scores = []
    for head_idx, head_pos_idx in enumerate(head_list):
        localization_scores.append(get_localization_score_single(
            positive_attributions, head_idx, head_pos_idx, scale=scale)[0])
    return torch.cat(localization_scores)


def plot_localization_scores_single(localization_scores, model, setting, exp, config, layer, scale=2):
    """
    Plots a given set of localization scores into a box plot

    :param localization_scores: Localization scores
    :type localization_scores: list
    :param model: Name of the model, for labelling
    :type model: str
    :param setting: Name of the setting, for labelling
    :type setting: str
    :param exp: Name of the attribution method, for labelling
    :type exp: str
    :param config: Name of the attribution configuration, for labelling
    :type config: str
    :param layer: Name of the layer evaluated on, for labelling
    :type layer: str
    :param scale: Grid dimension N, defaults to 2
    :type scale: int, optional
    :return: Figure and axis from Matplotlib
    :rtype: tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 10))
    method = "{}, {}, {}, {}, {}".format(model, setting, exp, config, layer)
    plot_data = [[method, x] for x in localization_scores]
    df = pd.DataFrame(plot_data, columns=["Method", "Localization Metric"])
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=df, x="Method", y="Localization Metric", ax=ax, showfliers=False)
    baseline_score = 1./(scale*scale)
    ax.axhline(baseline_score, linestyle="--")
    ax.axhline(1.0, linestyle="--")
    ax.set_ylim(-0.1, 1.1)
    return fig, ax
