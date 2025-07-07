import torch

from ripe import utils

log = utils.get_pylogger(__name__)


class DatasetCombinator:
    """Combines multiple datasets into one. Length of the combined dataset is the length of the
    longest dataset. Shorter datasets are looped over.

    Args:
        datasets: List of datasets to combine.
        mode: How to sample from the datasets. Can be either "uniform" or "weighted".
            In "uniform" mode, each dataset is sampled with equal probability.
            In "weighted" mode, each dataset is sampled with probability proportional to its length.
    """

    def __init__(self, datasets, mode="uniform", weights=None):
        self.datasets = datasets

        names_datasets = [type(ds).__name__ for ds in self.datasets]
        self.lengths = [len(ds) for ds in datasets]

        if mode == "weighted":
            self.probs_datasets = [length / sum(self.lengths) for length in self.lengths]
        elif mode == "uniform":
            self.probs_datasets = [1 / len(self.datasets) for _ in self.datasets]
        elif mode == "custom":
            assert weights is not None, "Weights must be provided in custom mode"
            assert len(weights) == len(datasets), "Number of weights must match number of datasets"
            assert sum(weights) == 1.0, "Weights must sum to 1"
            self.probs_datasets = weights
        else:
            raise ValueError(f"Unknown mode {mode}")

        log.info("Got the following datasets: ")

        for name, length, prob in zip(names_datasets, self.lengths, self.probs_datasets):
            log.info(f"{name} with {length} samples and probability {prob}")
        log.info(f"Total number of samples: {sum(self.lengths)}")

        self.num_samples = max(self.lengths)

        self.dataset_dist = torch.distributions.Categorical(probs=torch.tensor(self.probs_datasets))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        positive_sample = idx % 2 == 0

        if positive_sample:
            dataset_idx = self.dataset_dist.sample().item()

            idx = torch.randint(0, self.lengths[dataset_idx], (1,)).item()
            while idx % 2 == 1:
                idx = torch.randint(0, self.lengths[dataset_idx], (1,)).item()

            return self.datasets[dataset_idx][idx]
        else:
            dataset_idx_1 = self.dataset_dist.sample().item()
            dataset_idx_2 = self.dataset_dist.sample().item()

            if dataset_idx_1 == dataset_idx_2:
                idx = torch.randint(0, self.lengths[dataset_idx_1], (1,)).item()
                while idx % 2 == 0:
                    idx = torch.randint(0, self.lengths[dataset_idx_1], (1,)).item()
                return self.datasets[dataset_idx_1][idx]

            else:
                idx_1 = torch.randint(0, self.lengths[dataset_idx_1], (1,)).item()
                idx_2 = torch.randint(0, self.lengths[dataset_idx_2], (1,)).item()

                sample_1 = self.datasets[dataset_idx_1][idx_1]
                sample_2 = self.datasets[dataset_idx_2][idx_2]

                sample = {
                    "label": False,
                    "src_path": sample_1["src_path"],
                    "trg_path": sample_2["trg_path"],
                    "src_image": sample_1["src_image"],
                    "trg_image": sample_2["trg_image"],
                    "src_mask": sample_1["src_mask"],
                    "trg_mask": sample_2["trg_mask"],
                    "homography": sample_2["homography"],
                }
                return sample
