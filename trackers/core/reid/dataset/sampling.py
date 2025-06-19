import random
from collections import defaultdict
from typing import Any, Dict, List, Union

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class PKSampler(Sampler):
    """
    A sampler that samples a batch of `num_identities * num_instances` identities from the dataset.

    Reference: Section 3.1 of the paper
    [ICE: Inter-instance Contrastive Encoding for Unsupervised Person Re-identification](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/hao_iccv2021.pdf)

    Args:
        data_source (Union[List[Dict[str, Any]], Dataset]): A list of dictionaries containing the keys
            "image_path", "entity_id", and "camera_id" or the corresponding `torch.utils.data.Dataset` instance.
        num_identities (int): The number of identities to sample.
        num_instances (int): The number of instances to sample for each identity.
    """  # noqa: E501

    def __init__(
        self,
        data_source: Union[List[Dict[str, Any]], Dataset],
        num_identities: int,
        num_instances: int,
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_identities = num_identities
        self.num_instances = num_instances
        self.batch_size = num_identities * num_instances

        self.index_dict = defaultdict(list)
        for index, item in enumerate(self.data_source):
            self.index_dict[item["entity_id"]].append(index)
        self.entity_ids = list(self.index_dict.keys())

        if len(self.entity_ids) < self.num_identities:
            raise ValueError(
                f"The number of identities in the dataset is less than the number of identities to sample: {len(self.entity_ids)} < {self.num_identities}"  # noqa: E501
            )

        self.num_samples = (
            len(self.entity_ids) // self.num_identities
        ) * self.batch_size

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __iter__(self):
        """
        Iterates over the dataset, yielding sample indices in a structured manner.
        The iterator groups samples by identity, ensuring each batch contains exactly
        `num_identities` different identities with `num_instances` samples each.

        Yields:
            int: Individual sample indices from the dataset, yielded in batches
                of size `num_identities * num_instances`.
        """
        random.shuffle(self.entity_ids)
        batch_indices = []
        # Partition the shuffled identities into chunks of size num_identities
        # If necessary, wrap around for the final incomplete chunk
        for i in range(0, len(self.entity_ids), self.num_identities):
            identities_chunk = self.entity_ids[i : i + self.num_identities]
            # If the final chunk is smaller than num_identities, wrap around
            if len(identities_chunk) < self.num_identities:
                remainder = self.num_identities - len(identities_chunk)
                identities_chunk += random.sample(self.entity_ids, remainder)
            # For each identity in this chunk, sample num_instances indices
            for entity_id in identities_chunk:
                indices = self.index_dict[entity_id]
                if len(indices) >= self.num_instances:
                    selected = random.sample(indices, self.num_instances)
                else:
                    # If fewer than K instances, sample with replacement
                    selected = random.choices(indices, k=self.num_instances)
                batch_indices.extend(selected)
            # Once we have P*K indices, yield them as a batch
            if len(batch_indices) == self.batch_size:
                for idx in batch_indices:
                    yield idx
                batch_indices = []
        if len(batch_indices) > 0:
            for idx in batch_indices:
                yield idx
