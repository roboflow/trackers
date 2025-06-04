from __future__ import annotations

from typing import Any, List, Dict, Optional, Union, Tuple

from PIL import Image
from supervision.dataset.utils import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class IdentityDataset(Dataset):
    """
    A dataset that provides each individual identity along with associated images and metadata as a sample.
    This dataset is useful for training models with cross-entropy loss.
    
    Args:
        identity_mappings (List[Dict[str, Any]]): A list of dictionaries mapping image paths to entity IDs.
        transforms (Optional[Compose]): A torchvision.transforms.Compose object to apply to the images.
    """
    def __init__(self, identity_mappings: List[Dict[str, Any]], transforms: Optional[Compose] = None):
        self.identity_mappings = identity_mappings
        self.transforms = transforms or ToTensor()
    
    def get_num_identities(self):
        """
        Returns the number of unique identities in the dataset.
        """
        identities = set()
        for item in self.identity_mappings:
            identity = item["entity_id"]
            identities.add(identity)
        return len(identities)

    def __len__(self):
        """
        Returns the total number of identities in the dataset.
        """
        return len(self.identity_mappings)

    def __getitem__(self, index) -> dict[str, Any]:
        """
        Retrieves a sample from the dataset which is a dictionary containing
        the keys "image_path", "entity_id", "camera_id", and "image" (tensor).
        
        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict[str, Any]: A dictionary containing the image and metadata for the sample.
        """
        item = self.identity_mappings[index]
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transforms(image)
        item["image"] = image
        return item

    def split(
        self,
        split_ratio: float = 0.8,
        random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
        shuffle: bool = True,
    ) -> Tuple[IdentityDataset, IdentityDataset]:
        train_identity_mappings, validation_identity_mappings = train_test_split(
            list(self.identity_mappings),
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )
        return (
            IdentityDataset(train_identity_mappings, self.transforms),
            IdentityDataset(validation_identity_mappings, self.transforms),
        )
