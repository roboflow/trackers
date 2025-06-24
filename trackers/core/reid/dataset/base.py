from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

from PIL import Image
from supervision.dataset.utils import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class IdentityDataset(Dataset):
    def __init__(
        self,
        data: list[Tuple[str, int, int]],
        transforms: Optional[Union[Callable, Compose]] = None,
    ):
        self.data = data
        self.transforms = transforms or Compose([ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_path, identity, camera_id = self.data[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return image, identity, camera_id

    def get_num_identities(self) -> int:
        identities = set()
        for items in self.data:
            identities.add(items[1])
        return len(identities)

    def split(
        self,
        split_ratio: float = 0.8,
        random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
        shuffle: bool = True,
    ) -> Tuple[IdentityDataset, IdentityDataset]:
        train_data, validation_data = train_test_split(
            data=self.data,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )
        return (
            IdentityDataset(train_data, transforms=self.transforms),
            IdentityDataset(validation_data, transforms=self.transforms),
        )
