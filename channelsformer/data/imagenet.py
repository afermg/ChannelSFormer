from typing import Callable, Optional
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from channelsformer.data.utils import get_image

class ImageNet(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.df = pd.read_parquet(
            data_path
        )
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_hwc = get_image(row["path"])  # np.ndarray

        if img_hwc is None:
            raise RuntimeError(f"Failed to load image from path: {row['path']}")

        sample = Image.fromarray(img_hwc)
        label = row["label"]

        if self.transform is not None:
            sample = self.transform(sample)

        # convert label of numpy.int64 to pytorch long tensor
        label = torch.tensor(label, dtype=torch.long)

        return sample, label
