import json
import random
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from ripe import utils
from ripe.data.data_transforms import Compose

log = utils.get_pylogger(__name__)


class DISK_Megadepth(Dataset):
    def __init__(
        self,
        root: str,
        max_scene_size: int,
        stage: str = "train",
        # condition: str = "rain",
        transforms: Optional[Callable] = None,
        positive_only: bool = False,
    ) -> None:
        self.root = root
        self.stage = stage
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

        if self.stage not in ["train"]:
            raise RuntimeError("Unknown option " + self.stage + " as training stage variable. Valid options: 'train'")

        json_path = self.root / "megadepth" / "dataset.json"
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        self.scenes = []

        for scene in json_data:
            self.scenes.append(Scene(self.root / "megadepth", json_data[scene], max_scene_size))

        self.tuples_per_scene = [len(scene) for scene in self.scenes]

        if positive_only:
            log.warning("Using only positive pairs!")

    def __len__(self) -> int:
        if self.positive_only:
            return sum(self.tuples_per_scene)
        return 2 * sum(self.tuples_per_scene)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Any = {}

        positive_sample = idx % 2 == 0 or self.positive_only
        if not self.positive_only:
            idx = idx // 2

        sample["label"] = positive_sample

        i_scene, i_image = self._get_scene_and_image_id_from_idx(idx)

        if positive_sample:
            sample["src_path"], sample["trg_path"] = self.scenes[i_scene][i_image]

            homography = torch.eye(3, dtype=torch.float32)
        else:
            sample["src_path"], _ = self.scenes[i_scene][i_image]

            i_scene_other, i_image_other = self._get_other_random_scene_and_image_id(i_scene)

            sample["trg_path"], _ = self.scenes[i_scene_other][i_image_other]

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

    def _get_scene_and_image_id_from_idx(self, idx: int) -> Tuple[int, int]:
        accumulated_tuples = accumulate(self.tuples_per_scene)

        if idx >= sum(self.tuples_per_scene):
            raise IndexError(f"Index {idx} out of bounds")

        idx_scene = None
        for i, accumulated_tuple in enumerate(accumulated_tuples):
            idx_scene = i
            if idx < accumulated_tuple:
                break

        idx_image = idx - sum(self.tuples_per_scene[:idx_scene])

        return idx_scene, idx_image

    def _get_other_random_scene_and_image_id(self, scene_id_to_exclude: int) -> Tuple[int, int]:
        possible_scene_ids = list(range(len(self.scenes)))
        possible_scene_ids.remove(scene_id_to_exclude)

        idx_scene = random.choice(possible_scene_ids)
        idx_image = random.randint(0, len(self.scenes[idx_scene]) - 1)

        return idx_scene, idx_image


class Scene:
    def __init__(self, root_path, scene_data: Dict[str, Any], max_size_scene) -> None:
        self.root_path = root_path
        self.image_path = Path(scene_data["image_path"])
        self.image_names = scene_data["images"]

        # randomly sample tuples
        if max_size_scene > 0:
            self.tuples = random.sample(scene_data["tuples"], min(max_size_scene, len(scene_data["tuples"])))

    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        idx_1, idx_2 = random.sample([0, 1, 2], 2)

        idx_1 = self.tuples[idx][idx_1]
        idx_2 = self.tuples[idx][idx_2]

        path_image_1 = str(self.root_path / self.image_path / self.image_names[idx_1])
        path_image_2 = str(self.root_path / self.image_path / self.image_names[idx_2])

        return path_image_1, path_image_2
