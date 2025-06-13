from trackers.core.sort.tracker import SORTTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.log import get_logger

__all__ = ["SORTTracker", "OCSORTTracker"]

logger = get_logger(__name__)

try:
    from trackers.core.deepsort.tracker import DeepSORTTracker
    from trackers.core.ocsort.tracker import OCSORTTracker
    from trackers.core.reid.model import ReIDModel

    __all__.extend(["DeepSORTTracker", "ReIDModel", "OCSORTTracker"])
except ImportError:
    logger.warning(
        "ReIDModel dependencies not installed. ReIDModel will not be available. "
        "Please run `pip install trackers[reid]` and try again."
    )
    pass
