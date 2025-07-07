from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from ripe import utils
from ripe.data.data_transforms import Compose
from ripe.utils.utils import get_other_random_id

log = utils.get_pylogger(__name__)


class ACDC(Dataset):
    def __init__(
        self,
        root: Path,
        stage: str = "train",
        condition: str = "rain",
        transforms: Optional[Callable] = None,
        positive_only: bool = False,
    ) -> None:
        self.root = root
        self.stage = stage
        self.condition = condition
        self.transforms = transforms
        self.positive_only = positive_only

        if isinstance(self.root, str):
            self.root = Path(self.root)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found at {self.root}")

        if transforms is None:
            self.transforms = Compose([])
        else:
            self.transforms = transforms

        if self.stage not in ["train", "val", "test", "pred"]:
            raise RuntimeError(
                "Unknown option "
                + self.stage
                + " as training stage variable. Valid options: 'train', 'val', 'test' and 'pred'"
            )

        if self.stage == "pred":  # prediction uses the test set
            self.stage = "test"

        if self.stage in ["val", "test", "pred"]:
            self.positive_only = True
            log.info(f"{self.stage} stage: Using only positive pairs!")

        weather_conditions = ["fog", "night", "rain", "snow"]

        if self.condition not in weather_conditions + ["all"]:
            raise RuntimeError(
                "Unknown option "
                + self.condition
                + " as weather condition variable. Valid options: 'fog', 'night', 'rain', 'snow' and 'all'"
            )

        self.weather_condition_query = weather_conditions if self.condition == "all" else [self.condition]

        self._read_sample_files()

        if positive_only:
            log.warning("Using only positive pairs!")
        log.info(f"Found {len(self.src_images)} source images and {len(self.trg_images)} target images.")

    def _read_sample_files(self):
        file_name_pattern_ref = "_ref_anon.png"
        file_name_pattern = "_rgb_anon.png"

        self.trg_images = []
        self.src_images = []

        for weather_condition in self.weather_condition_query:
            rgb_files = sorted(
                list(self.root.glob("rgb_anon/" + weather_condition + "/" + self.stage + "/**/*" + file_name_pattern)),
                key=lambda i: i.stem[:21],
            )

            src_images = sorted(
                list(
                    self.root.glob(
                        "rgb_anon/" + weather_condition + "/" + self.stage + "_ref" + "/**/*" + file_name_pattern_ref
                    )
                ),
                key=lambda i: i.stem[:21],
            )

            self.trg_images += rgb_files
            self.src_images += src_images

    def __len__(self) -> int:
        if self.positive_only:
            return len(self.trg_images)
        return 2 * len(self.trg_images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Any = {}

        positive_sample = (idx % 2 == 0) or (self.positive_only)
        if not self.positive_only:
            idx = idx // 2

        sample["label"] = positive_sample

        if positive_sample:
            sample["src_path"] = str(self.src_images[idx])
            sample["trg_path"] = str(self.trg_images[idx])

            assert self.src_images[idx].stem[:21] == self.trg_images[idx].stem[:21], (
                f"Source and target image mismatch: {self.src_images[idx]} vs {self.trg_images[idx]}"
            )

            src_img = read_image(sample["src_path"])
            trg_img = read_image(sample["trg_path"])

            homography = torch.eye(3, dtype=torch.float32)
        else:
            sample["src_path"] = str(self.src_images[idx])
            idx_other = get_other_random_id(idx, len(self) // 2)
            sample["trg_path"] = str(self.trg_images[idx_other])

            assert self.src_images[idx].stem[:21] != self.trg_images[idx_other].stem[:21], (
                f"Source and target image match for negative sample: {self.src_images[idx]} vs {self.trg_images[idx_other]}"
            )

            src_img = read_image(sample["src_path"])
            trg_img = read_image(sample["trg_path"])

            homography = torch.zeros((3, 3), dtype=torch.float32)

        src_img = src_img / 255.0
        trg_img = trg_img / 255.0

        _, H, W = src_img.shape

        src_mask = torch.ones((1, H, W), dtype=torch.uint8)
        trg_mask = torch.ones((1, H, W), dtype=torch.uint8)

        if self.transforms:
            src_img, trg_img, src_mask, trg_mask, _ = self.transforms(src_img, trg_img, src_mask, trg_mask, homography)

        sample["src_image"] = src_img
        sample["trg_image"] = trg_img
        sample["src_mask"] = src_mask.to(torch.bool)
        sample["trg_mask"] = trg_mask.to(torch.bool)
        sample["homography"] = homography

        return sample
