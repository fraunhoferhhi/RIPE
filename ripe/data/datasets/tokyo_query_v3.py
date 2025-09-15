import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from ripe.data.data_transforms import Compose


class TokyoQueryV3(Dataset):
    def __init__(
        self, root: str, stage: str = "train", transforms: Optional[Callable] = None, positive_only: bool = False
    ) -> None:
        if stage != "train":
            raise ValueError("TokyoQueryV3 only supports the 'train' stage.")

        self.root = Path(root)
        self.transforms = transforms if transforms else Compose([])
        self.positive_only = positive_only

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found at {self.root}")

        self.triplets = self._load_triplets()

    def _load_triplets(self) -> List[List[Path]]:
        images = sorted([p for p in self.root.glob("*.jpg")], key=lambda x: int(x.stem))
        triplets = [images[i : i + 3] for i in range(0, len(images), 3)]
        return triplets

    def __len__(self) -> int:
        if self.positive_only:
            return len(self.triplets) * 3
        return len(self.triplets) * 6

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Dict[str, Any] = {}

        positive_sample = idx % 2 == 0 or self.positive_only
        if not self.positive_only:
            idx = idx // 2

        sample["label"] = positive_sample

        triplet_idx = idx // 3
        pair_idx = idx % 3

        if positive_sample:
            sample["src_path"] = str(self.triplets[triplet_idx][pair_idx])
            sample["trg_path"] = str(self.triplets[triplet_idx][(pair_idx + 1) % 3])
            homography = torch.eye(3, dtype=torch.float32)
        else:
            sample["src_path"] = str(self.triplets[triplet_idx][pair_idx])
            other_triplet_idx = random.choice([i for i in range(len(self.triplets)) if i != triplet_idx])
            sample["trg_path"] = str(self.triplets[other_triplet_idx][random.randint(0, 2)])
            homography = torch.zeros((3, 3), dtype=torch.float32)

        src_img = read_image(sample["src_path"]) / 255.0
        trg_img = read_image(sample["trg_path"]) / 255.0

        _, H_src, W_src = src_img.shape
        _, H_trg, W_trg = trg_img.shape

        src_mask = torch.ones((1, H_src, W_src), dtype=torch.uint8)
        trg_mask = torch.ones((1, H_trg, W_trg), dtype=torch.uint8)

        if self.transforms:
            src_img, trg_img, src_mask, trg_mask, _ = self.transforms(src_img, trg_img, src_mask, trg_mask, homography)

        sample["src_image"] = src_img
        sample["trg_image"] = trg_img
        sample["src_mask"] = src_mask.to(torch.bool)
        sample["trg_mask"] = trg_mask.to(torch.bool)
        sample["homography"] = homography

        return sample
