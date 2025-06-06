from trackers.log import get_logger

logger = get_logger(__name__)

try:
    from trackers.core.reid.dataset.base import TripletsDataset
    from trackers.core.reid.dataset.market_1501 import get_market1501_triplets_dataset
    from trackers.core.reid.dataset.sampling import PKSampler
    from trackers.core.reid.model import ReIDModel

    __all__ = [
        "PKSampler",
        "ReIDModel",
        "TripletsDataset",
        "get_market1501_triplets_dataset",
    ]
except ImportError:
    logger.warning(
        "ReIDModel dependencies not installed. ReIDModel will not be available. "
        "Please run `pip install trackers[reid]` and try again."
    )
    pass
