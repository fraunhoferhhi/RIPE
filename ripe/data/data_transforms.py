import collections
import collections.abc

import kornia.geometry as KG
import numpy as np
import torch
from torchvision.transforms import functional as TF


class Compose:
    """Composes several transforms together. The transforms are applied in the order they are passed in.
    Args:        transforms (list): A list of transforms to be applied.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, src, trg, src_mask, trg_mask, h):
        for t in self.transforms:
            src, trg, src_mask, trg_mask, h = t(src, trg, src_mask, trg_mask, h)

        return src, trg, src_mask, trg_mask, h


class Transform:
    """Base class for all transforms. It provides a method to apply a transformation function to the input images and masks.
    Args:
        src (torch.Tensor): The source image tensor.
        trg (torch.Tensor): The target image tensor.
        src_mask (torch.Tensor): The source image mask tensor.
        trg_mask (torch.Tensor): The target image mask tensor.
        h (torch.Tensor): The homography matrix tensor.
    Returns:
        tuple: A tuple containing the transformed source image, the transformed target image, the transformed source mask,
        the transformed target mask and the updated homography matrix.
    """

    def __init__(self):
        pass

    def apply_transform(self, src, trg, src_mask, trg_mask, h, transfrom_function):
        src, trg, src_mask, trg_mask, h = transfrom_function(src, trg, src_mask, trg_mask, h)
        return src, trg, src_mask, trg_mask, h


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, src, trg, src_mask, trg_mask, h):
        return self.apply_transform(src, trg, src_mask, trg_mask, h, self.transform_function)

    def transform_function(self, src, trg, src_mask, trg_mask, h):
        src = TF.normalize(src, mean=self.mean, std=self.std)
        trg = TF.normalize(trg, mean=self.mean, std=self.std)
        return src, trg, src_mask, trg_mask, h


class ResizeAndPadWithHomography(Transform):
    def __init__(self, target_size_longer_side=768):
        self.target_size = target_size_longer_side

    def __call__(self, src, trg, src_mask, trg_mask, h):
        return self.apply_transform(src, trg, src_mask, trg_mask, h, self.transform_function)

    def transform_function(self, src, trg, src_mask, trg_mask, h):
        src_w, src_h = src.shape[-1], src.shape[-2]
        trg_w, trg_h = trg.shape[-1], trg.shape[-2]

        # Resizing logic for both images
        scale_src, new_src_w, new_src_h = self.compute_resize(src_w, src_h)
        scale_trg, new_trg_w, new_trg_h = self.compute_resize(trg_w, trg_h)

        # Resize both images
        src_resized = TF.resize(src, [new_src_h, new_src_w])
        trg_resized = TF.resize(trg, [new_trg_h, new_trg_w])

        src_mask_resized = TF.resize(src_mask, [new_src_h, new_src_w])
        trg_mask_resized = TF.resize(trg_mask, [new_trg_h, new_trg_w])

        # Pad the resized images to be square (768x768)
        src_padded, src_padding = self.apply_padding(src_resized, new_src_w, new_src_h)
        trg_padded, trg_padding = self.apply_padding(trg_resized, new_trg_w, new_trg_h)

        src_mask_padded, _ = self.apply_padding(src_mask_resized, new_src_w, new_src_h)
        trg_mask_padded, _ = self.apply_padding(trg_mask_resized, new_trg_w, new_trg_h)

        # Update the homography matrix
        h = self.update_homography(h, scale_src, src_padding, scale_trg, trg_padding)

        return src_padded, trg_padded, src_mask_padded, trg_mask_padded, h

    def compute_resize(self, w, h):
        if w > h:
            scale = self.target_size / w
            new_w = self.target_size
            new_h = int(h * scale)
        else:
            scale = self.target_size / h
            new_h = self.target_size
            new_w = int(w * scale)
        return scale, new_w, new_h

    def apply_padding(self, img, new_w, new_h):
        pad_w = (self.target_size - new_w) // 2
        pad_h = (self.target_size - new_h) // 2
        padding = [
            pad_w,
            pad_h,
            self.target_size - new_w - pad_w,
            self.target_size - new_h - pad_h,
        ]
        img_padded = TF.pad(img, padding, fill=0)  # Zero-pad
        return img_padded, padding

    def update_homography(self, h, scale_src, padding_src, scale_trg, padding_trg):
        # Create the scaling matrices
        scale_matrix_src = np.array([[scale_src, 0, 0], [0, scale_src, 0], [0, 0, 1]])
        scale_matrix_trg = np.array([[scale_trg, 0, 0], [0, scale_trg, 0], [0, 0, 1]])

        # Create the padding translation matrices
        pad_matrix_src = np.array([[1, 0, padding_src[0]], [0, 1, padding_src[1]], [0, 0, 1]])
        pad_matrix_trg = np.array([[1, 0, -padding_trg[0]], [0, 1, -padding_trg[1]], [0, 0, 1]])

        # Update the homography: apply scaling and translation
        h_updated = (
            pad_matrix_trg
            @ scale_matrix_trg
            @ h.numpy()
            @ np.linalg.inv(scale_matrix_src)
            @ np.linalg.inv(pad_matrix_src)
        )

        return torch.from_numpy(h_updated).float()


class Resize(Transform):
    def __init__(self, output_size, edge_divisible_by=None, side="long", antialias=True):
        self.output_size = output_size
        self.edge_divisible_by = edge_divisible_by
        self.side = side
        self.antialias = antialias

    def __call__(self, src, trg, src_mask, trg_mask, h):
        return self.apply_transform(src, trg, src_mask, trg_mask, h, self.transform_function)

    def transform_function(self, src, trg, src_mask, trg_mask, h):
        new_size_src = self.get_new_image_size(src)
        new_size_trg = self.get_new_image_size(trg)

        src, T_src = self.resize(src, new_size_src)
        trg, T_trg = self.resize(trg, new_size_trg)

        src_mask, _ = self.resize(src_mask, new_size_src)
        trg_mask, _ = self.resize(trg_mask, new_size_trg)

        h = torch.from_numpy(T_trg @ h.numpy() @ T_src).float()

        return src, trg, src_mask, trg_mask, h

    def resize(self, img, size):
        h, w = img.shape[-2:]

        img = KG.transform.resize(
            img,
            size,
            side=self.side,
            antialias=self.antialias,
            align_corners=None,
            interpolation="bilinear",
        )

        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        T = np.diag([scale[0].item(), scale[1].item(), 1])

        return img, T

    def get_new_image_size(self, img):
        h, w = img.shape[-2:]

        if isinstance(self.output_size, collections.abc.Iterable):
            assert len(self.output_size) == 2
            return tuple(self.output_size)
        if self.output_size is None:  # keep the original size, but possibly make it divisible by edge_divisible_by
            size = (h, w)
        else:
            side_size = self.output_size
            aspect_ratio = w / h
            if self.side not in ("short", "long", "vert", "horz"):
                raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{self.side}'")
            if self.side == "vert":
                size = side_size, int(side_size * aspect_ratio)
            elif self.side == "horz":
                size = int(side_size / aspect_ratio), side_size
            elif (self.side == "short") ^ (aspect_ratio < 1.0):
                size = side_size, int(side_size * aspect_ratio)
            else:
                size = int(side_size / aspect_ratio), side_size

        if self.edge_divisible_by is not None:
            df = self.edge_divisible_by
            size = list(map(lambda x: int(x // df * df), size))
        return size
