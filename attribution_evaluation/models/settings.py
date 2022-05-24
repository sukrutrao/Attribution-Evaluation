import torch
from .. import utils


def get_setting(setting_name):
    """
    Maps setting names to classes.

    :param setting_name: Name of the setting
    :type setting_name: str
    :return: Class for the setting
    :rtype: GridContainerBase
    """
    settings_dict = {"GridPG": GridPGContainer,
                     "DiFull": DiFullContainer, "DiPart": DiPartContainer}
    return settings_dict[setting_name]


def eval_only_corners(setting_name):
    """
    Maps setting names to whether the setting evaluates only at the top-left and bottom-right corners, as opposed to the entire grid.

    :param setting_name: Name of the setting
    :type setting_name: str
    :return: Whether the setting evaluates only at the top-left and bottom-right corners of the grid
    :rtype: bool
    """
    settings_dict = {"GridPG": False,
                     "DiFull": True, "DiPart": True}
    return settings_dict[setting_name]


class GridContainerBase(torch.nn.Module):
    """
    Base class for the grid evaluation settings.
    """

    def __init__(self, model, scale=2):
        """
        Constructor.

        :param model: Model to evaluate on.
        :type model: ModelBase
        :param scale: Scale parameter n in the nxn grid to use for evaluation, defaults to 2
        :type scale: int, optional
        """
        super(GridContainerBase, self).__init__()
        assert scale >= 1 and isinstance(scale, int)
        self.model = model
        self.scale = scale
        self.num_heads = scale * scale

    def forward(self, x, start_layer=None):
        raise NotImplementedError

    def get_intermediate_activations(self, x, output_head_idx=0, end_layer=None):
        raise NotImplementedError

    def predict(self, x, *kwargs):
        """
        Runs the model and returns softmax activations.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :return: Softmax activations.
        :rtype: torch.Tensor
        """
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class GridPGContainer(GridContainerBase):
    """
    Evaluation on the GridPG setting.
    """

    def __init__(self, model, scale=2):
        """
        Constructor.

        :param model: Model to evaluate on.
        :type model: ModelBase
        :param scale: scale parameter n in the nxn grid to use for evaluation, defaults to 2
        :type scale: int, optional
        """
        super(GridPGContainer, self).__init__(model, scale)
        self.single_head = True  # Since there is only one classification head
        self.model.enable_classifier_kernel()  # Used by VGG

    def forward(self, x, output_head_idx=0, start_layer=None):
        """
        Runs the model on the GridPG setting and returns output logits.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :param output_head_idx: Index of the classification head to evaluate for, defaults to 0. Ignored in the GridPG setting, since there is only one classification head.
        :type output_head_idx: int, optional
        :param start_layer: Convolutional layer ID to start the forward pass from, defaults to None. When None, start from the input image.
        :type start_layer: int, optional
        :return: Output logits.
        :rtype: torch.Tensor
        """
        features = self.model.get_features(
            x, start_layer=start_layer, end_layer=None)
        pool = self.model.get_pool(features)
        logits = self.model.get_logits(pool)
        return logits

    @torch.no_grad()
    def get_intermediate_activations(self, x, end_layer=None):
        """
        Returns intermediate features from the model at a specified convolutional layer.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :param end_layer: Convolutional layer ID at which features are to be returned, defaults to None. When None, return the final feature map.
        :type end_layer: int, optional
        :return: Intermediate features.
        :rtype: torch.Tensor
        """
        return self.model.get_features(x, start_layer=None, end_layer=end_layer)


class DiFullContainer(GridContainerBase):
    """
    Evaluation on the DiFull setting.
    """

    def __init__(self, model, scale=2):
        """
        Constructor.

        :param model: Model to evaluate on.
        :type model: ModelBase
        :param scale: scale parameter n in the nxn grid to use for evaluation, defaults to 2
        :type scale: int, optional
        """
        super(DiFullContainer, self).__init__(model, scale)
        self.single_head = False
        self.model.disable_classifier_kernel()

    def forward(self, x, output_head_idx=0, start_layer=None):
        """
        Runs the model on the DiFull setting and returns output logits.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :param output_head_idx: Index of the classification head to evaluate for, defaults to 0.
        :type output_head_idx: int, optional
        :param start_layer: Convolutional layer ID to start the forward pass from, defaults to None. When None, start from the input image.
        :type start_layer: int, optional
        :return: Output logits.
        :rtype: torch.Tensor
        """
        assert output_head_idx >= 0 and output_head_idx < self.num_heads

        # Find the coordinates to slice the grid cell at index output_head_idx from the input
        y_coord, x_coord, height, width = utils.get_augmentation_range(
            x.shape, self.scale, output_head_idx)
        features = self.model.get_features(
            x[:, :, y_coord:y_coord + height, x_coord:x_coord + width], start_layer=start_layer, end_layer=None)
        pool = self.model.get_pool(features)
        logits = self.model.get_logits(pool)
        return logits

    @torch.no_grad()
    def get_intermediate_activations(self, x, end_layer=None):
        """
        Returns intermediate features from the model at a specified convolutional layer.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :param end_layer: Convolutional layer ID at which features are to be returned, defaults to None. When None, return the final feature map.
        :type end_layer: int, optional
        :return: Intermediate features.
        :rtype: torch.Tensor
        """
        intermediate_activations = []

        # Get intermediate activations for each grid cell by passing them separately through the model, and stitch them together at the end
        for row_idx in range(self.scale):
            row_activations = []
            for col_idx in range(self.scale):
                head_idx = row_idx * self.scale + col_idx
                y_coord, x_coord, height, width = utils.get_augmentation_range(
                    x.shape, self.scale, head_idx)
                row_activations.append(self.model.get_features(
                    x[:, :, y_coord:y_coord + height, x_coord:x_coord + width], start_layer=None, end_layer=end_layer))
            intermediate_activations.append(torch.cat(row_activations, dim=3))
        return torch.cat(intermediate_activations, dim=2)


class DiPartContainer(GridContainerBase):
    """
    Evaluation on the DiPart setting.
    """

    def __init__(self, model, scale=2):
        """
        Constructor.

        :param model: Model to evaluate on.
        :type model: ModelBase
        :param scale: scale parameter n in the nxn grid to use for evaluation, defaults to 2
        :type scale: int, optional
        """
        super(DiPartContainer, self).__init__(model, scale)
        self.single_head = False
        self.model.disable_classifier_kernel()

    def forward(self, x, output_head_idx=0, start_layer=None):
        """
        Runs the model on the DiPart setting and returns output logits.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :param output_head_idx: Index of the classification head to evaluate for, defaults to 0.
        :type output_head_idx: int, optional
        :param start_layer: Convolutional layer ID to start the forward pass from, defaults to None. When None, start from the input image.
        :type start_layer: int, optional
        :return: Output logits.
        :rtype: torch.Tensor
        """
        assert output_head_idx >= 0 and output_head_idx < self.num_heads
        features = self.model.get_features(
            x, start_layer=start_layer, end_layer=None)
        y_coord, x_coord, height, width = utils.get_augmentation_range(
            features.shape, self.scale, output_head_idx)
        pool = self.model.get_pool(
            features[:, :, y_coord:y_coord + height, x_coord:x_coord + width])
        logits = self.model.get_logits(pool)
        return logits

    @torch.no_grad()
    def get_intermediate_activations(self, x, end_layer=None):
        """
        Returns intermediate features from the model at a specified convolutional layer.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :param end_layer: Convolutional layer ID at which features are to be returned, defaults to None. When None, return the final feature map.
        :type end_layer: int, optional
        :return: Intermediate features.
        :rtype: torch.Tensor
        """
        return self.model.get_features(x, start_layer=None, end_layer=end_layer)
