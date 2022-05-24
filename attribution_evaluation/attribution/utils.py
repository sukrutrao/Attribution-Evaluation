import numpy as np
import torch
from functools import wraps


def limit_n_images(func):
    def wrapper(mod, img, target, *args, **kwargs):
        if hasattr(mod, "max_imgs_bs") and len(img) > mod.max_imgs_bs:
            batch_size = mod.max_imgs_bs
            return torch.cat([
                func(mod,
                     img[idx * batch_size: (idx + 1) * batch_size],
                     target[idx * batch_size: (idx + 1) * batch_size], *args, **kwargs)
                for idx in range(int(np.ceil(len(img) / batch_size)))], dim=0)
        return func(mod, img, target, *args, **kwargs)

    return wrapper
