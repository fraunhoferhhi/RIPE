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
from ripe.utils.image_utils import Camera, cameras2F

log = utils.get_pylogger(__name__)


class DISK_IMW(Dataset):
    def __init__(
        self,
        root: str,
        stage: str = "val",
        # condition: str = "rain",
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.stage = stage
        self.transforms = transforms

        if isinstance(self.root, str):
            self.root = Path(self.root)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found at {self.root}")

        if transforms is None:
            self.transforms = Compose([])
        else:
            self.transforms = transforms

        if self.stage not in ["val"]:
            raise RuntimeError("Unknown option " + self.stage + " as training stage variable. Valid options: 'train'")

        json_path = self.root / "imw2020-val" / "dataset.json"
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        self.scenes = []

        for scene in json_data:
            self.scenes.append(Scene(self.root / "imw2020-val", json_data[scene]))

        self.tuples_per_scene = [len(scene) for scene in self.scenes]

    def __len__(self) -> int:
        return sum(self.tuples_per_scene)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Any = {}

        i_scene, i_image = self._get_scene_and_image_id_from_idx(idx)

        sample["src_path"], sample["trg_path"], path_calib_src, path_calib_trg = self.scenes[i_scene][i_image]

        cam_src = Camera.from_calibration_file(path_calib_src)
        cam_trg = Camera.from_calibration_file(path_calib_trg)

        F = self.get_F(cam_src, cam_trg)
        s2t_R, s2t_T = self.get_relative_pose(cam_src, cam_trg)

        src_img = read_image(sample["src_path"]) / 255.0
        trg_img = read_image(sample["trg_path"]) / 255.0

        _, H_src, W_src = src_img.shape
        _, H_trg, W_trg = trg_img.shape

        src_mask = torch.ones((1, H_src, W_src), dtype=torch.uint8)
        trg_mask = torch.ones((1, H_trg, W_trg), dtype=torch.uint8)

        H = torch.eye(3)
        if self.transforms:
            src_img, trg_img, src_mask, trg_mask, _ = self.transforms(src_img, trg_img, src_mask, trg_mask, H)

        # check if transformations in self.transforms. Only Normalize is allowed
        for t in self.transforms.transforms:
            if t.__class__.__name__ not in ["Normalize", "Resize"]:
                raise ValueError(f"Transform {t.__class__.__name__} not allowed in DISK_IMW dataset")

        sample["src_image"] = src_img
        sample["trg_image"] = trg_img
        sample["orig_size_src"] = (H_src, W_src)
        sample["orig_size_trg"] = (H_trg, W_trg)
        sample["src_mask"] = src_mask.to(torch.bool)
        sample["trg_mask"] = trg_mask.to(torch.bool)
        sample["F"] = F
        sample["s2t_R"] = s2t_R
        sample["s2t_T"] = s2t_T
        sample["src_camera"] = cam_src
        sample["trg_camera"] = cam_trg

        return sample

    def get_relative_pose(self, cam_src: Camera, cam_trg: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
        R = cam_trg.R @ cam_src.R.T
        T = cam_trg.t - R @ cam_src.t

        return R, T

    def get_F(self, cam_src: Camera, cam_trg: Camera) -> torch.Tensor:
        F = cameras2F(cam_src, cam_trg)

        return F

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
    def __init__(self, root_path, scene_data: Dict[str, Any]) -> None:
        self.root_path = root_path
        self.image_path = Path(scene_data["image_path"])
        self.calib_path = Path(scene_data["calib_path"])
        self.image_names = scene_data["images"]
        self.tuples = scene_data["tuples"]

    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx_1 = self.tuples[idx][0]
        idx_2 = self.tuples[idx][1]

        path_image_1 = str(self.root_path / self.image_path / self.image_names[idx_1]) + ".jpg"
        path_image_2 = str(self.root_path / self.image_path / self.image_names[idx_2]) + ".jpg"
        path_calib_1 = str(self.root_path / self.calib_path / ("calibration_" + self.image_names[idx_1])) + ".h5"
        path_calib_2 = str(self.root_path / self.calib_path / ("calibration_" + self.image_names[idx_2])) + ".h5"

        return path_image_1, path_image_2, path_calib_1, path_calib_2
