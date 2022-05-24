import torch
from captum.attr import LayerAttribution


def interpolate_attributions(attributions, img_dims, interpolate_mode="bilinear"):
    """
    Upscales attributions to a specific dimension

    :param attributions: Attributions
    :type attributions: torch.Tensor of the shape (B, K, 1, H, W), where B is the batch size, K is the number of grid cells per image, H is the image height, and W is the image width
    :param img_dims: Dimensions to upscale to
    :type img_dims: tuple. Dimensions (h, w), where h is the height of the grid cell, and w is the width of the grid cell.
    :param interpolate_mode: Interpolation mode, defaults to "bilinear"
    :type interpolate_mode: str, optional
    :return: Upscaled attributions
    :rtype: torch.Tensor of the shape (B, K, 1, img_dims[0], img_dims[1])
    """
    original_shape = attributions.shape
    attributions = LayerAttribution.interpolate(attributions.flatten(
        start_dim=0, end_dim=1), img_dims, interpolate_mode=interpolate_mode).reshape(original_shape[:3] + img_dims)
    return attributions


def get_positive_attributions(attributions):
    """
    Zeros out negative attributions and returns a new tensor with only positive attributions

    :param attributions: Attributions
    :type attributions: torch.Tensor of the shape (B, K, 1, H, W), where B is the batch size, K is the number of grid cells per image, H is the image height, and W is the image width
    :return: Positive attributions
    :rtype: torch.Tensor of the shape (B, K, 1, H, W), where B is the batch size, K is the number of grid cells per image, H is the image height, and W is the image width 
    """
    positive_attributions = attributions.clone()
    positive_attributions[torch.where(positive_attributions < 0)] = 0
    return positive_attributions


def get_sorted_localization_scores(localization_scores, attributions_inside):
    """
    Sorts localization scores in descending order, breaking ties by favouring data points with higher sum of positive attributions inside the target grid cell

    :param localization_scores: Localization scores
    :type localization_scores: list
    :param attributions_inside: Sum of attributions inside the target grid cell for each data point
    :type attributions_inside: list
    :return: Sorted localization scores, sum of attributions inside, and original position indices
    :rtype: tuple
    """
    localization_tuples = [(score, inside, idx) for idx, (score, inside)
                           in enumerate(zip(localization_scores, attributions_inside))]
    localization_tuples.sort(key=lambda x: (x[0], x[1]), reverse=True)
    localization_tuples = torch.tensor(localization_tuples)
    return localization_tuples[:, 0], localization_tuples[:, 1], localization_tuples[:, 2].long()
