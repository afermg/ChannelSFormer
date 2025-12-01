from typing import Callable, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from channelsformer.data.utils import get_image

DATASET_MEAN = [0.1830, 0.0766, 0.1717, 0.1555, 0.1914, 0.2958, 0.2961, 0.2957]
DATASET_STD = [0.1547, 0.1514, 0.1604, 0.1372, 0.1631, 0.0196, 0.0216, 0.0214]


def load_meta_data():
    PLATE_TO_ID = {"BR00116991": 0, "BR00116993": 1, "BR00117000": 2}
    FIELD_TO_ID = dict(zip([str(i) for i in range(1, 10)], range(9)))
    WELL_TO_ID = {}
    for i in range(16):
        for j in range(1, 25):
            well_loc = f"{chr(ord('A') + i)}{j:02d}"
            WELL_TO_ID[well_loc] = len(WELL_TO_ID)

    WELL_TO_LBL = {}
    # map the well location to the perturbation label
    base_path = "s3://insitro-research-2023-context-vit/jumpcp/platemap_and_metadata"
    PLATE_MAP = {
        "compound": f"{base_path}/JUMP-Target-1_compound_platemap.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_platemap.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_platemap.tsv",
    }
    META_DATA = {
        "compound": f"{base_path}/JUMP-Target-1_compound_metadata.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_metadata.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_metadata.tsv",
    }

    for perturbation in PLATE_MAP.keys():
        df_platemap = pd.read_parquet(PLATE_MAP[perturbation])
        df_metadata = pd.read_parquet(META_DATA[perturbation])
        df = df_metadata.merge(df_platemap, how="inner", on="broad_sample")

        if perturbation == "compound":
            target_name = "target"
        else:
            target_name = "gene"

        codes, uniques = pd.factorize(df[target_name])
        codes += 1  # set none (neg control) to id 0
        assert min(codes) == 0
        print(f"{target_name} has {len(uniques)} unique values")
        WELL_TO_LBL[perturbation] = dict(zip(df["well_position"], codes))

    return PLATE_TO_ID, FIELD_TO_ID, WELL_TO_ID, WELL_TO_LBL


class Jumpcp(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        transform_init_fn: Optional[Callable] = None,
        transform_params=None,
        # split: train, valid, test, train_valid, all
        split: Optional[str] = "test",
    ):
        super().__init__()

        self.transform = transform
        self.transform_init_fn = transform_init_fn
        self.transform_params = transform_params
        df = pd.read_parquet(data_path)
        self.split = split
        self.df = self.get_split(df, split)

        self.data_path = list(self.df["path"])
        self.data_id = list(self.df["ID"])
        self.well_loc = list(self.df["well_loc"])

        self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data()

        self.perturbation_type = "compound"  # perturbation_list[0]

    def __len__(self) -> int:
        return len(self.df)

    def get_transform_x(self, x) -> None:
        if self.transform is None:
            if self.transform_init_fn is not None:
                self.transform = self.transform_init_fn(**self.transform_params)
                return self.transform(x)
            else:
                return x
        else:
            return self.transform(x)

    def get_split(self, df, split_name, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(0.6 * m)
        validate_end = int(0.2 * m) + train_end

        if split_name == "train":
            return df.iloc[perm[:train_end]]
            # return df.iloc[perm[:4096]] # for debugging only
        elif split_name == "valid":
            return df.iloc[perm[train_end:validate_end]]
            # return df.iloc[perm[train_end:train_end + 4096]]  # for debugging only
        elif split_name == "test":
            return df.iloc[perm[validate_end:]]
        elif split_name == "train_valid":
            return df.iloc[perm[:validate_end]]
        elif split_name == "all":
            return df
        else:
            raise ValueError("Unknown split")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["path"]
        img_hwc = get_image(file_path)  # np.ndarray

        if img_hwc is None:
            raise RuntimeError(f"Failed to load image from path: {file_path}")

        sample = torch.from_numpy(img_hwc).float()
        with torch.no_grad():
            sample = self.get_transform_x(sample)

        if len(sample.shape) == 4:
            sample = sample.squeeze(0)

        # Check if well location exists in label mapping
        if self.well_loc[idx] not in self.well2lbl[self.perturbation_type]:
            raise KeyError(
                f"Well location '{self.well_loc[idx]}' not found in "
                f"label mapping for perturbation type '{self.perturbation_type}'"
            )

        label = self.well2lbl[self.perturbation_type][self.well_loc[idx]]

        return sample, label