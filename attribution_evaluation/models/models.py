import torch
import torchvision


def get_model(model_name):
    """
    Maps model names to classes.

    :param model_name: Name of the model
    :type model_name: str
    :return: Class for the model
    :rtype: ModelBase
    """
    models_dict = {"vgg11": VGG11Model, "resnet18": Resnet18Model}
    return models_dict[model_name]


class ModelBase(torch.nn.Module):
    """
    Base class for a model. To be used with a Torchvision model, adds additional functions for evaluating on the grid setting.
    """

    def __init__(self):
        super(ModelBase, self).__init__()
        self.use_classifier_kernel = False
        self.classifier_kernel_created = False

    def get_features(self, x, start_layer=None, end_layer=None):
        raise NotImplementedError

    def get_pool(self, x):
        raise NotImplementedError

    def get_logits(self, x):
        raise NotImplementedError

    def enable_classifier_kernel(self):
        pass

    def disable_classifier_kernel(self):
        pass


class VGG11Model(ModelBase):
    """
    Augments torchvision.models.vgg11 with additional functionality for evaluating on the grid setting.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(VGG11Model, self).__init__()
        self.base_model = torchvision.models.vgg11(pretrained=True)
        self.layer_map = {"Input": 0, "Middle": 5, "Final": 8}

        # Store list of indexes for evaluating, right after the ReLU following a convolutional layer
        # First element of -1 denotes evaluation at the input
        self.conv_layer_ids = [-1] + [idx for (idx, layer) in enumerate(
            self.base_model.features.children()) if isinstance(layer, torch.nn.ReLU)]

    def enable_classifier_kernel(self):
        """
        Enables the use of a convolutional kernel to replace the VGG11 classifier. To be used in the GridPG setting. Creates the classifier kernel when called the first time.
        """
        if not self.classifier_kernel_created:
            self._create_classifier_kernel()
        self.use_classifier_kernel = True

    def disable_classifier_kernel(self):
        """
        Disables the use of a convolutional kernel to replace the VGG11 classifier in the GridPG setting.
        """
        self.use_classifier_kernel = False

    def _create_classifier_kernel(self):
        """
        Defines a classifier kernel to replace the VGG11 classifier in the GridPG setting. Equivalent to sliding the original VGG11 classifier with stride=1 across the feature grid.
        """

        # Reformulate the three linear layers in the classifier as 1x1 convolutions
        self.classifier_conv1 = torch.nn.Conv2d(
            512, 4096, kernel_size=7, padding=0)
        self.classifier_conv2 = torch.nn.Conv2d(
            4096, 4096, kernel_size=1, padding=0)
        self.classifier_conv3 = torch.nn.Conv2d(
            4096, 1000, kernel_size=1, padding=0)
        self.avgpool2d = torch.nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier_conv1.weight.data = self.base_model.classifier[0].weight.reshape(
            (4096, 512, 7, 7))
        self.classifier_conv1.bias.data = self.base_model.classifier[0].bias
        self.classifier_conv2.weight.data = self.base_model.classifier[3].weight.reshape(
            (4096, 4096, 1, 1))
        self.classifier_conv2.bias.data = self.base_model.classifier[3].bias
        self.classifier_conv3.weight.data = self.base_model.classifier[6].weight.reshape(
            (1000, 4096, 1, 1))
        self.classifier_conv3.bias.data = self.base_model.classifier[6].bias

        self.classifier_kernel_created = True

    def get_features(self, x, start_layer=None, end_layer=None):
        """
        Returns features for an input from the model evaluated on a specified range of layers.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :param start_layer: Convolutional layer ID at which to begin forward pass, defaults to None. When None, begin from the input.
        :type start_layer: int, optional
        :param end_layer: Convolutional layer ID at which to end forward pass, defaults to None. When None, end at the final feature layer.
        :type end_layer: int, optional
        :return: Feature map at the specified end layer.
        :rtype: torch.Tensor
        """
        start_layer = 0 if start_layer is None else self.conv_layer_ids[start_layer] + 1
        end_layer = len(
            self.base_model.features) if end_layer is None else self.conv_layer_ids[end_layer] + 1
        for idx in range(start_layer, end_layer):
            x = self.base_model.features[idx](x)
        return x

    def get_pool(self, x):
        """
        Pools the feature maps. To be used before sending features to the classifier.

        :param x: Feature maps.
        :type x: torch.Tensor
        :return: Pooled feature maps.
        :rtype: torch.Tensor
        """

        # For VGG11, feature outputs are already 7x7 for a 224x224 input. Even in the three grid evaluation settings, we only pass 7x7 blocks through the classifier. So no explicit pooling step is necessary.
        return x

    def get_logits(self, x):
        """
        Returns the logits from the model given pooled features.

        :param x: Pooled feature maps.
        :type x: torch.Tensor
        :return: Output logits.
        :rtype: torch.Tensor
        """
        if self.use_classifier_kernel:
            x = self.classifier_conv1(x)
            x = self.base_model.classifier[1](x)
            x = self.base_model.classifier[2](x)
            x = self.classifier_conv2(x)
            x = self.base_model.classifier[4](x)
            x = self.base_model.classifier[5](x)
            x = self.classifier_conv3(x)
            x = self.avgpool2d(x).squeeze(2).squeeze(2)
            return x

        return self.base_model.classifier(x.flatten(1))


class Resnet18Model(ModelBase):
    """
    Augments torchvision.models.resnet18 with additional functionality for evaluating on the grid setting.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(Resnet18Model, self).__init__()
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.layer_map = {"Input": 0, "Middle": 3, "Final": 5}

        self.all_layers = list(self.base_model.named_children())
        self.feature_layers = self.all_layers[:-2]
        self.pool_layer = self.all_layers[-2][1]
        self.classifier = self.all_layers[-1][1]
        self.conv_layer_ids = [-1, 2, 4, 5, 6, 7]

    def get_features(self, x, start_layer=None, end_layer=None):
        """
        Returns features for an input from the model evaluated on a specified range of layers.

        :param x: Input image or intermediate activations.
        :type x: torch.Tensor
        :param start_layer: Convolutional layer ID at which to begin forward pass, defaults to None. When None, begin from the input.
        :type start_layer: int, optional
        :param end_layer: Convolutional layer ID at which to end forward pass, defaults to None. When None, end at the final feature layer.
        :type end_layer: int, optional
        :return: Feature map at the specified end layer.
        :rtype: torch.Tensor
        """
        start_layer = 0 if start_layer is None else self.conv_layer_ids[start_layer] + 1
        end_layer = len(
            self.feature_layers) if end_layer is None else self.conv_layer_ids[end_layer] + 1
        for idx in range(start_layer, end_layer):
            x = self.feature_layers[idx][1](x)
        return x

    def get_pool(self, x):
        """
        Pools the feature maps. To be used before sending features to the classifier.

        :param x: Feature maps.
        :type x: torch.Tensor
        :return: Pooled feature maps.
        :rtype: torch.Tensor
        """
        return self.pool_layer(x)

    def get_logits(self, x):
        """
        Returns the logits from the model given pooled features.

        :param x: Pooled feature maps.
        :type x: torch.Tensor
        :return: Output logits.
        :rtype: torch.Tensor
        """
        return self.classifier(x.flatten(1))
