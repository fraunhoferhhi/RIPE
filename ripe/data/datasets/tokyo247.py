import os
import random
from glob import glob
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from ripe import utils
from ripe.data.data_transforms import Compose

log = utils.get_pylogger(__name__)


class Tokyo247(Dataset):
    def __init__(
        self,
        root: str,
        stage: str = "train",
        transforms: Optional[Callable] = None,
        positive_only: bool = False,
    ):
        if stage != "train":
            raise ValueError("Tokyo247Dataset only supports the 'train' stage.")

        # check if the root directory exists
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Directory {root} does not exist.")

        self.root_dir = root
        self.transforms = transforms if transforms is not None else Compose([])
        self.positive_only = positive_only

        self.image_paths = []
        self.positive_pairs = []

        # Collect images grouped by location folder
        self.locations = {}
        for location_rough in sorted(os.listdir(self.root_dir)):
            location_rough_path = os.path.join(self.root_dir, location_rough)

            # check if the location_rough_path is a directory
            if not os.path.isdir(location_rough_path):
                continue

            for location_fine in sorted(os.listdir(location_rough_path)):
                location_fine_path = os.path.join(self.root_dir, location_rough, location_fine)

                if os.path.isdir(location_fine_path):
                    images = sorted(
                        glob(os.path.join(location_fine_path, "*.png")),
                        key=lambda i: int(i[-7:-4]),
                    )
                    if len(images) >= 12:
                        self.locations[location_fine] = images
                        self.image_paths.extend(images)

        # Generate positive pairs
        for _, images in self.locations.items():
            for i in range(len(images) - 1):
                self.positive_pairs.append((images[i], images[i + 1]))
            self.positive_pairs.append((images[-1], images[0]))

        if positive_only:
            log.warning("Using only positive pairs!")

        log.info(f"Found {len(self.positive_pairs)} image pairs.")

    def __len__(self):
        if self.positive_only:
            return len(self.positive_pairs)
        return 2 * len(self.positive_pairs)

    def __getitem__(self, idx):
        sample: Any = {}

        positive_sample = (idx % 2 == 0) or (self.positive_only)
        if not self.positive_only:
            idx = idx // 2

        sample["label"] = positive_sample

        if positive_sample:  # Positive pair
            img1_path, img2_path = self.positive_pairs[idx]

            assert os.path.dirname(img1_path) == os.path.dirname(img2_path), (
                f"Source and target image mismatch: {img1_path} vs {img2_path}"
            )

            homography = torch.eye(3, dtype=torch.float32)
        else:  # Negative pair
            img1_path = random.choice(self.image_paths)
            img2_path = random.choice(self.image_paths)

            # Ensure images are from different folders
            esc = 0
            while os.path.dirname(img1_path) == os.path.dirname(img2_path):
                img2_path = random.choice(self.image_paths)

                esc += 1
                if esc > 100:
                    raise RuntimeError("Could not find a negative pair.")

            assert os.path.dirname(img1_path) != os.path.dirname(img2_path), (
                f"Source and target image match for negative pair: {img1_path} vs {img2_path}"
            )

            homography = torch.zeros((3, 3), dtype=torch.float32)

        sample["src_path"] = img1_path
        sample["trg_path"] = img2_path

        # Load images
        src_img = read_image(sample["src_path"]) / 255.0
        trg_img = read_image(sample["trg_path"]) / 255.0

        _, H_src, W_src = src_img.shape
        _, H_trg, W_trg = src_img.shape

        src_mask = torch.ones((1, H_src, W_src), dtype=torch.uint8)
        trg_mask = torch.ones((1, H_trg, W_trg), dtype=torch.uint8)

        # Apply transformations
        if self.transforms:
            src_img, trg_img, src_mask, trg_mask, _ = self.transforms(src_img, trg_img, src_mask, trg_mask, homography)

        sample["src_image"] = src_img
        sample["trg_image"] = trg_img
        sample["src_mask"] = src_mask.to(torch.bool)
        sample["trg_mask"] = trg_mask.to(torch.bool)
        sample["homography"] = homography

        return sample
