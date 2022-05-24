import torch
import matplotlib.pyplot as plt
from . import visualization_captum
import math
from . import utils
from . import localization
import matplotlib
matplotlib.use("Agg")


def visualize_aggatt(attributions, head_idx=0, head_pos_idx=0, img_dims=(224, 224), scale=2, percentile=0.5, display_negative=False, plt_fig_axis=None, fig_size=2, bin_positions_list=[0.0, 0.02, 0.05, 0.5, 0.95, 0.98, 1.0]):
    """
    Generates AggAtt visualizations from a set of attributions for a specific classification head

    :param attributions: Attributions
    :type attributions: torch.Tensor of the shape (B, K, 1, H, W)
    :param head_idx: Index of the classification head in the attribution tensor, defaults to 0
    :type head_idx: int, optional
    :param head_pos_idx: Index of the position of the grid cell in the grid, defaults to 0
    :type head_pos_idx: int, optional
    :param img_dims: Dimensions of each grid cell, defaults to (224, 224)
    :type img_dims: tuple, optional
    :param scale: Grid dimension N, defaults to 2
    :type scale: int, optional
    :param percentile: Percentile of attributions to clamp in each bin, for handling outliers, defaults to 0.5
    :type percentile: float, optional
    :param display_negative: Flag to enable displaying negative attributions, defaults to False
    :type display_negative: bool, optional
    :param plt_fig_axis: Matplotlib figure and axis to plot on, defaults to None
    :type plt_fig_axis: tuple, optional
    :param fig_size: Scale parameter for the output figure size, defaults to 2
    :type fig_size: int, optional
    :param bin_positions_list: Fractions of the total number of attributions at which to demarcate AggAtt bins, defaults to [0.0, 0.02, 0.05, 0.5, 0.95, 0.98, 1.0]
    :type bin_positions_list: list, optional
    :return: Matplotlib figure and axis
    :rtype: tuple
    """
    grid_img_dims = tuple([scale * dim for dim in img_dims])
    interpolated_attributions = utils.interpolate_attributions(
        attributions, grid_img_dims)
    positive_attributions = utils.get_positive_attributions(
        interpolated_attributions)
    num_attributions = len(interpolated_attributions)
    _validate_bin_positions_list(bin_positions_list, num_attributions)
    num_bins = len(bin_positions_list) - 1
    localization_scores, attributions_inside = localization.get_localization_score_single(
        positive_attributions, head_idx, head_pos_idx, scale)
    _, _, sorted_indices = utils.get_sorted_localization_scores(
        localization_scores, attributions_inside)
    if display_negative:
        display_attributions = interpolated_attributions[sorted_indices]
    else:
        display_attributions = positive_attributions[sorted_indices]
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        plt_fig, plt_axis = plt.subplots(
            1, num_bins, figsize=(fig_size * num_bins, fig_size))
    bin_maps = []
    for aggatt_bin in range(num_bins):
        bin_size = int(num_attributions *
                       (bin_positions_list[aggatt_bin + 1] - bin_positions_list[aggatt_bin]))
        bin_start_idx = int(bin_positions_list[aggatt_bin] * num_attributions)
        bin_end_idx = int(
            bin_positions_list[aggatt_bin + 1] * num_attributions)
        bin_mean = display_attributions[bin_start_idx:bin_end_idx, head_idx].sum(
            dim=0) / bin_size
        bin_maps.append(bin_mean)
    all_bins = torch.stack(bin_maps).cpu().numpy()
    scale_factor = visualization_captum.threshold_value_based(
        all_bins, 100 - percentile)
    for aggatt_bin in range(num_bins):
        visualization_captum.visualize_image_attr_custom(bin_maps[aggatt_bin].permute(1, 2, 0).cpu().numpy(
        ), None, method='heat_map', sign='all', show_colorbar=False, use_pyplot=False, plt_fig_axis=(plt_fig, plt_axis[aggatt_bin]), scale_factor=scale_factor)
        plt_axis[aggatt_bin].add_patch(_get_bounding_box(
            head_pos_idx, img_dims=img_dims, scale=scale))
    return plt_fig, plt_axis


def visualize_examples(attributions, images, num_examples=10, head_idx=0, head_pos_idx=0, img_dims=(224, 224), scale=2, percentile=0.5, display_negative=False, plt_fig_axis=None, fig_size=2, bin_positions_list=[0.0, 0.02, 0.05, 0.5, 0.95, 0.98, 1.0]):
    """
    Plots examples from each AggAtt bin from a set of attributions for a specific classification head

    :param attributions: Attributions
    :type attributions: torch.Tensor of the shape (B, K, 1, H, W)
    :param images: Images from which attributions where generated, in the same order
    :type images: torch.Tensor of the shape (B, C, H, W)
    :param num_examples: Number of examples to plot, defaults to 10
    :type num_examples: int, optional
    :param head_idx: Index of the classification head in the attribution tensor, defaults to 0
    :type head_idx: int, optional
    :param head_pos_idx: Index of the position of the grid cell in the grid, defaults to 0
    :type head_pos_idx: int, optional
    :param img_dims: Dimensions of each grid cell, defaults to (224, 224)
    :type img_dims: tuple, optional
    :param scale: Grid dimension N, defaults to 2
    :type scale: int, optional
    :param percentile: Percentile of attributions to clamp for each example, for handling outliers, defaults to 0.5
    :type percentile: float, optional
    :param display_negative: Flag to enable displaying negative attributions, defaults to False
    :type display_negative: bool, optional
    :param plt_fig_axis: Matplotlib figure and axis to plot on, defaults to None
    :type plt_fig_axis: tuple, optional
    :param fig_size: Scale parameter for the output figure size, defaults to 2
    :type fig_size: int, optional
    :param bin_positions_list: Fractions of the total number of attributions at which to demarcate AggAtt bins, defaults to [0.0, 0.02, 0.05, 0.5, 0.95, 0.98, 1.0]
    :type bin_positions_list: list, optional
    :return: Matplotlib figure and axis
    :rtype: tuple
    """
    grid_img_dims = tuple([scale * dim for dim in img_dims])
    interpolated_attributions = utils.interpolate_attributions(
        attributions, grid_img_dims)
    positive_attributions = utils.get_positive_attributions(
        interpolated_attributions)
    num_attributions = len(interpolated_attributions)
    _validate_bin_positions_list(bin_positions_list, num_attributions)
    num_bins = len(bin_positions_list) - 1
    localization_scores, attributions_inside = localization.get_localization_score_single(
        positive_attributions, head_idx, head_pos_idx, scale)
    _, _, sorted_indices = utils.get_sorted_localization_scores(
        localization_scores, attributions_inside)
    if display_negative:
        display_attributions = interpolated_attributions[sorted_indices]
    else:
        display_attributions = positive_attributions[sorted_indices]
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        plt_fig, plt_axis = plt.subplots(num_examples, 2*num_bins, figsize=(
            2*fig_size * num_bins, fig_size * num_examples))
    if num_examples == 1:
        plt_axis = plt_axis.reshape(1, -1)

    for aggatt_bin in range(num_bins):
        bin_start_idx = int(bin_positions_list[aggatt_bin] * num_attributions)
        bin_end_idx = int(
            bin_positions_list[aggatt_bin + 1] * num_attributions)
        bin_median_idx = (bin_start_idx + bin_end_idx) // 2
        for example_idx in range(num_examples):
            offset_idx = ((example_idx + 1) // 2) * \
                int(math.pow(-1, example_idx))
            attribution_idx = bin_median_idx + offset_idx
            assert attribution_idx >= bin_start_idx and attribution_idx < bin_end_idx, "At least one bin contains fewer than {} attributions, cannot display {} examples".format(
                num_examples, num_examples)
            scale_factor = visualization_captum.threshold_value_based(
                display_attributions[attribution_idx, head_idx].cpu().numpy(), 100 - percentile)
            visualization_captum.visualize_image_attr_custom(None, images[sorted_indices][attribution_idx].permute(1, 2, 0).cpu().numpy(
            ), method='original_image', sign='all', show_colorbar=False, use_pyplot=False, plt_fig_axis=(plt_fig, plt_axis[example_idx, aggatt_bin*2]), scale_factor=scale_factor)
            visualization_captum.visualize_image_attr_custom(display_attributions[attribution_idx, head_idx].permute(1, 2, 0).cpu().numpy(
            ), None, method='heat_map', sign='all', show_colorbar=False, use_pyplot=False, plt_fig_axis=(plt_fig, plt_axis[example_idx, aggatt_bin*2+1]), scale_factor=scale_factor)
            plt_axis[example_idx, aggatt_bin *
                     2].add_patch(_get_bounding_box(head_pos_idx, img_dims=img_dims, scale=scale))
            plt_axis[example_idx, aggatt_bin*2 +
                     1].add_patch(_get_bounding_box(head_pos_idx, img_dims=img_dims, scale=scale))
    return plt_fig, plt_axis


def _validate_bin_positions_list(bin_positions_list, attributions_size):
    """
    Validation checks for the bin_positions_list parameter, to check if the specified bins can be created, if the attributions can be evenly divided into them, and if each bin contains at least one attribution

    :param bin_positions_list: Fractions of the total number of attributions at which to demarcate AggAtt bins
    :type bin_positions_list: list
    :param attributions_size: Total number of attributions
    :type attributions_size: int
    """
    for idx in range(len(bin_positions_list) - 1):
        assert bin_positions_list[idx] >= 0 and bin_positions_list[idx] <= 1, "AggAtt bin fractions should lie in [0,1]"
        assert bin_positions_list[idx +
                                  1] > bin_positions_list[idx], "AggAtt bin at position {} is empty or of negative size".format(idx)
        bin_size = (bin_positions_list[idx + 1] - bin_positions_list[idx]) * attributions_size
        assert math.isclose(bin_size - round(bin_size), 0, abs_tol=1e-6
                            ), "Number of attributions cannot be evenly divided into AggAtt bin at position {}".format(idx)


def _get_bounding_box(head_pos_idx, img_dims=(224, 224), scale=2):
    """
    Generates a bounding box to plot on AggAtt bins or examples based on the grid cell being visualized

    :param head_pos_idx: Index of the position of the grid cell in the grid
    :type head_pos_idx: int
    :param img_dims: Dimensions of each grid cell, defaults to (224, 224)
    :type img_dims: tuple, optional
    :param scale: Grid dimension N, defaults to 2
    :type scale: int, optional
    :return: Bounding box to plot
    :rtype: matplotlib.patches.Rectangle
    """
    row_idx = head_pos_idx // scale
    col_idx = head_pos_idx % scale
    box_x = col_idx * img_dims[1] - 0.5
    box_y = row_idx * img_dims[0] - 0.5
    box = matplotlib.patches.Rectangle(
        (box_x, box_y), img_dims[0] - 0.25, img_dims[1] - 0.25, edgecolor='b', facecolor='none')
    return box
