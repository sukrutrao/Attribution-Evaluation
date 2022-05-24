# Copyright notice and license for Captum, most code below modifies functions from it
# BSD 3-Clause License

# Copyright (c) 2019, PyTorch team
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from captum.attr import visualization as viz
import matplotlib
matplotlib.use('Agg')


def threshold_value_based(values, percentile):
    """
    Performs thresholding to clamp outliers for better visualization. Unlike in Captum, threshold by clamping the top percentile % of values, inside of the top percentile % of total attributions, since the former is more stable if an attribution method gives very few extreme valued attributions.

    :param values: Aboslute value of attributions
    :type values: np.array
    :param percentile: Percentile of attributions to clamp
    :type percentile: float
    :return: Threshold at which to clamp
    :rtype: float
    """
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    threshold_id = int(len(sorted_vals) * 0.01 * percentile)
    return sorted_vals[threshold_id]


def visualize_image_attr_custom(attr, original_image=None, method='heat_map', sign='absolute_value', plt_fig_axis=None, outlier_perc=2, cmap=None, alpha_overlay=0.5, show_colorbar=False, title=None, fig_size=(6, 6), use_pyplot=True, scale_factor=None, color_original_image=False):
    """
    Modifies captum.attr.visualization.visualize_image_attr by using value based outlier clamping and by allowing the normalization factor to be specified by the caller. The latter helps in using a common normalization factor for a series of images, as in AggAtt.
    """
    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(figsize=fig_size)
        else:
            plt_fig = matplotlib.figure.Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots()

    if original_image is not None:
        if np.max(original_image) <= 1.0:
            original_image = viz._prepare_image(original_image * 255)
    else:
        assert (
            viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.heat_map
        ), "Original Image must be provided for any visualization other than heatmap."

    # Remove ticks and tick labels from plot.
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])
    plt_axis.grid(b=False)

    heat_map = None
    # Show original image
    if viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.original_image:
        if len(original_image.shape) > 2 and original_image.shape[2] == 1:
            original_image = np.squeeze(original_image, axis=2)
        plt_axis.imshow(original_image)
    else:
        # Choose appropriate signed attributions and normalize.
        if scale_factor != 0:
            norm_attr = viz._normalize_scale(attr, scale_factor)
        else:
            norm_attr = attr.copy()

        # Set default colormap and bounds based on sign.
        if viz.VisualizeSign[sign] == viz.VisualizeSign.all:
            default_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
            vmin, vmax = -1, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.positive:
            default_cmap = "Greens"
            vmin, vmax = 0, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.negative:
            default_cmap = "Reds"
            vmin, vmax = 0, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.absolute_value:
            default_cmap = "Blues"
            vmin, vmax = 0, 1
        else:
            raise AssertionError("Visualize Sign type is not valid.")
        cmap = cmap if cmap is not None else default_cmap

        # Show appropriate image visualization.
        if viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.heat_map:
            heat_map = plt_axis.imshow(
                norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
        elif (
            viz.ImageVisualizationMethod[method]
            == viz.ImageVisualizationMethod.blended_heat_map
        ):
            if color_original_image:
                plt_axis.imshow(original_image)
            else:
                plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
            heat_map = plt_axis.imshow(
                norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay
            )
        elif viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.masked_image:
            assert viz.VisualizeSign[sign] != viz.VisualizeSign.all, (
                "Cannot display masked image with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                viz._prepare_image(
                    original_image * np.expand_dims(norm_attr, 2))
            )
        elif viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.alpha_scaling:
            assert viz.VisualizeSign[sign] != viz.VisualizeSign.all, (
                "Cannot display alpha scaling with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                np.concatenate(
                    [
                        original_image,
                        viz._prepare_image(
                            np.expand_dims(norm_attr, 2) * 255),
                    ],
                    axis=2,
                )
            )
        else:
            raise AssertionError("Visualize Method type is not valid.")

    # Add colorbar. If given method is not a heatmap and no colormap is relevant,
    # then a colormap axis is created and hidden. This is necessary for appropriate
    # alignment when visualizing multiple plots, some with heatmaps and some
    # without.
    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes(
            "bottom", size="5%", pad=0.1)
        if heat_map:
            plt_fig.colorbar(
                heat_map, orientation="horizontal", cax=colorbar_axis)
        else:
            colorbar_axis.axis("off")
    if title:
        plt_axis.set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis
