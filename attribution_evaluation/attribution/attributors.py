from .configs import attributor_configs
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, GuidedBackprop, InputXGradient, Saliency, LayerAttribution
from skimage.transform import resize
from tqdm import tqdm
import os
from .utils import limit_n_images
import sys


def get_attributor(model_setting, attributor_name, config_name):
    """
    Maps the names and configurations to attributors for a specific model and setting

    :param model_setting: Evaluation setting containing the model to evaluate on
    :type model_setting: GridContainerBase
    :param attributor_name: Name of the attribution method
    :type attributor_name: str
    :param config_name: Name of the attribution method configuration
    :type config_name: str
    :return: Attributor object
    :rtype: AttributorBase
    """
    attributor_map = {
        "Grad": Grad,
        "GB": GB,
        "IntGrad": IntGrad,
        "IxG": IxG,
        "GradCam": GradCam,
        "GradCamPlusPlus": GradCamPlusPlus,
        "AblationCam": AblationCam,
        "ScoreCam": ScoreCam,
        "LayerCam": LayerCam,
        "Occlusion": Occlusion,
        "RISE": RISE,
    }
    return attributor_map[attributor_name](model_setting, **attributor_configs[attributor_name][config_name])


class AttributorContainer:
    """
    Container to evaluate an attribution method on a model on a specific classification head at a specific layer
    """

    def __init__(self, model_setting, base_exp, base_config):
        """
        Constructor

        :param model_setting: Setting object containing the model to evaluate on
        :type model_setting: GridContainerBase
        :param base_exp: Attribution method to evaluate on
        :type base_exp: str
        :param base_config: Attribution configuration to evaluate on
        :type base_config: str
        """
        self.model_setting = model_setting
        self.base_attributor = get_attributor(
            self.model_setting, base_exp, base_config)

    def attribute(self, img, target, output_head_idx=0, conv_layer_idx=0, **kwargs):
        """
        Runs the attribution method on a batch of images on the model in the prescribed setting on a specific layer, and if applicable, for a specific classification head

        :param img: Images to obtain attributions for
        :type img: torch.Tensor of the shape (B, C, H, W), where B is the batch size, C is the number of channels, H is the image height, and W is the image width
        :param target: Output logits from which to obtain attributions
        :type target: torch.Tensor of the shape (B, 1)
        :param output_head_idx: Classification head to evaluate on, defaults to 0. Required for the DiFull and DiPart settings. Classification heads for a N x N grid are counted zero-indexed row-wise from the top-left corner to the bottom-right corner.
        :type output_head_idx: int, optional
        :param conv_layer_idx: Layer to evaluate on, defaults to 0. A value of zero is equivalent to evaluating at the input.
        :type conv_layer_idx: int, optional
        :return: Attributions
        :rtype: torch.Tensor of the shape (B, 1, 1, H, W)
        """
        if isinstance(conv_layer_idx, str):
            conv_layer_idx = self.model_setting.model.layer_map[conv_layer_idx]
        assert conv_layer_idx >= 0 and conv_layer_idx < len(
            self.model_setting.model.conv_layer_ids)
        features = self.model_setting.get_intermediate_activations(img,
                                                                   end_layer=conv_layer_idx)
        if self.base_attributor.use_original_img:
            attrs = self.base_attributor.attribute(features, target=target, original_img=img,
                                                   additional_forward_args=(output_head_idx, conv_layer_idx), **self.base_attributor.configs, **kwargs)
        else:
            attrs = self.base_attributor.attribute(features, target=target, additional_forward_args=(
                output_head_idx, conv_layer_idx), **self.base_attributor.configs, **kwargs)
        attrs = attrs.sum(dim=1, keepdim=True).float()
        return attrs.detach()

    def attribute_selection(self, img, target, output_head_idx=0, conv_layer_idx=0, **kwargs):
        """
        Runs the attribution method on a batch of images for multiple targets on the model in the prescribed setting on a specific layer, and if applicable, for a specific classification head

        :param img: Images to obtain attributions for
        :type img: torch.Tensor of the shape (B, C, H, W), where B is the batch size, C is the number of channels, H is the image height, and W is the image width
        :param target: Output logits from which to obtain attributions
        :type target: torch.Tensor of the shape (B, K), where K is the number of targets per image.
        :param output_head_idx: Classification head to evaluate on, defaults to 0. Required for the DiFull and DiPart settings. Classification heads for a N x N grid are counted zero-indexed row-wise from the top-left corner to the bottom-right corner.
        :type output_head_idx: int, optional
        :param conv_layer_idx: Layer to evaluate on, defaults to 0. A value of zero is equivalent to evaluating at the input.
        :type conv_layer_idx: int, optional
        :return: Attributions
        :rtype: torch.Tensor of the shape (B, K, 1, H, W)
        """
        out = []
        for tgt_idx in range(target.shape[1]):
            out.append(self.attribute(img, target=target[:, tgt_idx].tolist(),
                                      output_head_idx=output_head_idx, conv_layer_idx=conv_layer_idx, **kwargs).detach().cpu())

        out = torch.stack(out, dim=1)
        return out


class AttributorBase:
    """
    Base class for attribution methods
    """

    def __init__(self, model_setting, **configs):
        self.model_setting = model_setting
        self.configs = configs
        self.use_original_img = False


class IntGrad(AttributorBase, IntegratedGradients):
    """
    Integrated Gradient attributions
    Reference: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model_setting, n_steps=20, internal_batch_size=1):
        AttributorBase.__init__(self,
                                model_setting, n_steps=n_steps, internal_batch_size=internal_batch_size)
        IntegratedGradients.__init__(self, self.model_setting)


class GB(AttributorBase, GuidedBackprop):
    """
    Guided Backprop attributions
    Reference: https://arxiv.org/abs/1412.6806
    """

    def __init__(self, model_setting, apply_abs=True):
        AttributorBase.__init__(self, model_setting)
        GuidedBackprop.__init__(self, self.model_setting)
        self.abs = apply_abs

    def attribute(self, img, target, additional_forward_args=None, **kwargs):
        attrs = super(GB, self).attribute(
            img, target, additional_forward_args=additional_forward_args)
        if self.abs:
            attrs = torch.abs(attrs)
        return attrs


class IxG(AttributorBase, InputXGradient):
    """
    InputxGradient attributions
    Reference: https://arxiv.org/abs/1704.02685
    """

    def __init__(self, model_setting):
        AttributorBase.__init__(self, model_setting)
        InputXGradient.__init__(self, self.model_setting)


class Grad(AttributorBase, Saliency):
    """
    Gradient attributions
    Reference: https://arxiv.org/abs/1312.6034
    """

    def __init__(self, model_setting, apply_abs=True, **configs):
        AttributorBase.__init__(self, model_setting, **configs)
        Saliency.__init__(self, self.model_setting)


class RISE(AttributorBase, nn.Module):
    """
    RISE attributions
    Reference: https://arxiv.org/abs/1806.07421
    """

    def __init__(self, model_setting, mask_path, batch_size=2, n=6000, s=6, p1=0.1):
        nn.Module.__init__(self)
        AttributorBase.__init__(self, model_setting)
        self.path_tmplt = os.path.join(mask_path, "masks{}_{}.npy")
        self.batch_size = batch_size
        self.max_imgs_bs = 1
        self.N = n
        self.s = s
        self.p1 = p1
        self.masks = None

    def generate_masks(self, savepath="masks.npy", input_size=None):
        print("Generating masks for", input_size)
        p1, s = self.p1, self.s
        if not os.path.isdir(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size
        grid = np.random.rand(self.N, s, s) < p1
        grid = grid.astype("float32")
        masks = np.empty((self.N, *input_size))
        for i in tqdm(range(self.N)):
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            masks[i, :, :] = resize((grid[i]), up_size, order=1, mode="reflect", anti_aliasing=False)[
                x:x + input_size[0], y:y + input_size[1]]

        masks = (masks.reshape)(*(-1, 1), *input_size)
        np.save(savepath, masks)

    def load_masks(self, filepath):
        if not os.path.exists(filepath):
            size = int(os.path.basename(filepath)[
                       len("masks") + len(str(self.N)) + len("_"):-len(".npy")])
            self.generate_masks(savepath=(filepath[:-4]),
                                input_size=(size, size))
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
        return self.masks

    @limit_n_images
    @torch.no_grad()
    def attribute(self, x, target, return_all=False, additional_forward_args=None):
        N = self.N
        _, _, H, W = x.size()
        if self.masks is None or self.masks.shape[(-1)] != H:
            self.masks = self.load_masks(
                self.path_tmplt.format(int(N), int(H)))
        stack = torch.mul(self.masks, x.data)
        p = []
        for i in range(0, N, self.batch_size):
            if additional_forward_args is not None:
                p.append((self.model_setting.predict)(
                    stack[i:min(i + self.batch_size, N)], *additional_forward_args))
            else:
                p.append(self.model_setting.predict(
                    stack[i:min(i + self.batch_size, N)]))

        p = torch.cat(p)
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, 1, H, W))
        sal = sal / N / self.p1
        if return_all:
            return sal
        return sal[int(target[0])][None]

    def attribute_selection(self, x, tgts, additional_forward_args=None):
        return self.attribute(x, tgts, return_all=True, additional_forward_args=additional_forward_args)[tgts]


class Occlusion(AttributorBase):
    """
    Occlusion attributions
    Reference: https://arxiv.org/abs/1311.2901
    """

    def __init__(self, model_setting, stride=32, ks=32, batch_size=8, only_positive=False):
        super().__init__(model_setting)
        self.masks = None
        self.participated = None
        self.n_part = None
        self.max_imgs_bs = 1
        if isinstance(stride, int):
            stride = (
                stride, stride)
        else:
            raise NotImplementedError
        self.stride = stride
        self.ks = ks
        self.batch_size = batch_size
        self.only_positive = only_positive

    def make_masks(self, img):
        stride = self.stride
        ks = self.ks
        total = (img.shape[(-1)] // stride[(-1)] + ks // stride[(-1)] - 1) * \
            (img.shape[(-2)] // stride[(-2)] + ks // stride[(-2)] - 1)
        strided_shape = (np.array(img.shape[-2:]) / np.array(stride)
                         ).astype(int) + ks // stride[(-1)] - 1
        if ks % 2 == 1:
            ks2 = (ks - 1) // 2
            off = 0
        else:
            ks2 = ks
            off = ks // stride[(-1)] - 1
        occlusion_masks = []
        if ks % 2 == 1:
            for idx in range(total):
                mask = torch.ones(img.shape[-2:])
                wpos, hpos = np.unravel_index(idx, shape=strided_shape)
                mask[max(0, (hpos + off) * stride[0] - ks2):min(img.shape[(-1)] + 1, hpos * stride[0] + ks2 + 1),
                     max(0, (wpos + off) * stride[1] - ks2):min(img.shape[(-1)] + 1, wpos * stride[1] + ks2 + 1)] = 0
                occlusion_masks.append(mask)

        else:
            for idx in range(total):
                mask = torch.ones(img.shape[-2:])
                wpos, hpos = np.unravel_index(idx, shape=strided_shape)
                mask[max(0, (hpos - off) * stride[0]):min(img.shape[(-1)], (hpos - off) * stride[0] + ks),
                     max(0, (wpos - off) * stride[1]):min(img.shape[(-1)], (wpos - off) * stride[1] + ks)] = 0
                occlusion_masks.append(mask)

        self.masks = torch.stack(occlusion_masks, dim=0)[:, None].cpu()

    @limit_n_images
    @torch.no_grad()
    def attribute(self, img, target, return_all=True, additional_forward_args=None):
        self.model_setting.zero_grad()
        batch_size = self.batch_size
        stride = self.stride
        if additional_forward_args is not None:
            org_out = (self.model_setting)(img, *additional_forward_args).cpu()
        else:
            org_out = self.model_setting(img).cpu()
        img = img.cpu()
        if self.masks is None or self.masks.shape[(-1)] != img.shape[(-1)]:
            self.make_masks(img)
            masks = self.masks
            participated = (masks - 1).abs()[:, 0]
            n_part = participated.view(
                masks.shape[0], -1).sum(1)[:, None, None, None]
            self.participated = participated[:, None]
            self.n_part = n_part
        masks = self.masks.cpu()
        masked_input = img * masks
        if additional_forward_args is not None:
            pert_out = torch.cat([(self.model_setting)(masked_input[idx * batch_size:(idx + 1) * batch_size].cuda(), *additional_forward_args).cpu() for idx in range(int(np.ceil(len(masked_input) / batch_size)))],
                                 dim=0)
        else:
            pert_out = torch.cat([self.model_setting(masked_input[idx * batch_size:(idx + 1) * batch_size].cuda()).cpu() for idx in range(int(np.ceil(len(masked_input) / batch_size)))],
                                 dim=0)
        diff = (
            org_out - pert_out).clamp(0) if self.only_positive else org_out - pert_out
        diff = diff[:, :, None, None]
        diff2 = diff[:, torch.tensor(target).flatten(), :, :]
        influence = self.participated * diff2 / self.n_part
        if return_all:
            return influence.sum(0, keepdim=True)
        return influence.sum(0, keepdim=True)[:, int(target)][:, None].cuda()

    def attribute_selection(self, img, targets):
        return self.attribute(img, targets, return_all=True).unsqueeze(2)


class GradCamBase(AttributorBase):
    """
    Base class for Grad-CAM like methods
    """

    def __init__(self, model_setting, pool_grads=True, only_positive_grads=False, use_higher_order_grads=False):
        super(GradCamBase, self).__init__(model_setting)
        self.pool_grads = pool_grads
        self.only_positive_grads = only_positive_grads
        self.use_higher_order_grads = use_higher_order_grads

    def attribute(self, img, target, additional_forward_args=None, **kwargs):
        img.requires_grad = True
        outs = self.model_setting(
            img, *additional_forward_args) if additional_forward_args is not None else self.model_setting(img)
        grads = torch.autograd.grad(outs[:, target].diag(), img,
                                    grad_outputs=(torch.ones_like(outs[:, target].diag())))[0]
        with torch.no_grad():
            if self.only_positive_grads:
                grads = torch.nn.functional.relu(grads)
            if self.pool_grads:
                if self.use_higher_order_grads:
                    weights = self._get_higher_order_grads(
                        img, grads, outs[:, target].diag())
                    prods = weights * img
                else:
                    prods = torch.mean(grads, dim=(2, 3),
                                       keepdim=True) * img
            else:
                prods = grads * img
            attrs = torch.nn.functional.relu(
                torch.sum(prods, axis=1, keepdim=True))
        return attrs.detach()

    def _get_higher_order_grads(self, conv_acts, grads, logits):
        alpha_num = torch.pow(grads, 2)
        alpha_den = 2 * torch.pow(grads, 2) + torch.pow(grads, 3) * conv_acts.sum(dim=(2,
                                                                                       3), keepdim=True)
        alpha_den = torch.where(
            alpha_den != 0.0, alpha_den, torch.ones_like(alpha_den))
        alpha = alpha_num / alpha_den
        weights = (torch.nn.functional.relu(torch.exp(logits).reshape(-1, 1, 1, 1).expand_as(grads) * grads) * alpha).sum(dim=(2,
                                                                                                                               3), keepdim=True)
        return weights


class GradCam(GradCamBase):
    """
    Grad-CAM attributions
    Reference: https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model_setting):
        super(GradCam, self).__init__(model_setting=model_setting, pool_grads=True,
                                      only_positive_grads=False,
                                      use_higher_order_grads=False)


class GradCamPlusPlus(GradCamBase):
    """
    Grad-CAM++ attributions
    Reference: https://arxiv.org/abs/1710.11063
    """

    def __init__(self, model_setting):
        super(GradCamPlusPlus, self).__init__(model_setting=model_setting, pool_grads=True,
                                              only_positive_grads=False,
                                              use_higher_order_grads=True)


class LayerCam(GradCamBase):
    """
    Layer-CAM attributions
    Reference: https://ieeexplore.ieee.org/document/9462463
    """

    def __init__(self, model_setting):
        super(LayerCam, self).__init__(model_setting=model_setting, pool_grads=False,
                                       only_positive_grads=True,
                                       use_higher_order_grads=False)


class AblationCam(AttributorBase):
    """
    Ablation-CAM attributions
    Reference: https://ieeexplore.ieee.org/document/9093360
    """

    def __init__(self, model_setting):
        super(AblationCam, self).__init__(model_setting)

    def attribute(self, img, target, additional_forward_args=None, **kwargs):
        with torch.no_grad():
            outs = self.model_setting(
                img, *additional_forward_args) if additional_forward_args is not None else self.model_setting(img)
            weights = torch.zeros((
                img.shape[0], img.shape[1], 1, 1)).cuda()
            for act_idx in range(img.shape[1]):
                acts_temp = torch.clone(img).detach()
                acts_temp[:, act_idx, :, :] = 0
                act_outs = (self.model_setting)(
                    acts_temp, *additional_forward_args) if additional_forward_args is not None else self.model_setting(acts_temp)
                original_preds = outs[:, target].diag()
                act_preds = act_outs[:, target].diag()
                weights[:, act_idx, 0, 0] = (
                    original_preds - act_preds) / original_preds

            prods = weights * img
            attrs = torch.nn.functional.relu(
                torch.sum(prods, axis=1, keepdim=True))
        return attrs.detach()


class ScoreCam(AttributorBase):
    """
    Score-CAM attributions
    Reference: https://arxiv.org/abs/1910.01279
    """

    def __init__(self, model_setting):
        super(ScoreCam, self).__init__(model_setting)
        self.use_original_img = True

    def attribute(self, img, target, original_img, additional_forward_args=None, **kwargs):
        with torch.no_grad():
            weights = torch.zeros((
                img.shape[0], img.shape[1], 1, 1)).cuda()
            for act_idx in range(img.shape[1]):
                upsampled_acts = LayerAttribution.interpolate(img[:, act_idx, :, :].unsqueeze(1),
                                                              (original_img.shape[2], original_img.shape[3]), interpolate_mode="bilinear").detach()
                min_acts = torch.min(upsampled_acts.view(-1, original_img.shape[2] * original_img.shape[3]),
                                     dim=1)[0].reshape(-1, 1, 1, 1)
                max_acts = torch.max(upsampled_acts.view(-1, original_img.shape[2] * original_img.shape[3]),
                                     dim=1)[0].reshape(-1, 1, 1, 1)
                normalized_acts = (upsampled_acts - min_acts) / \
                    (max_acts - min_acts)
                mod_imgs = original_img * normalized_acts
                mod_outs = self.model_setting(mod_imgs).detach()
                weights[:, act_idx, 0, 0] = torch.nn.functional.softmax(mod_outs, dim=1)[:,
                                                                                         target].diag()

            prods = weights * img
            attrs = torch.nn.functional.relu(
                torch.sum(prods, axis=1, keepdim=True))
        return attrs.detach()
