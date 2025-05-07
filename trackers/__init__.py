from trackers.core.sort.tracker import SORTTracker
from trackers.log import get_logger

__all__ = ["SORTTracker"]

logger = get_logger(__name__)

try:
    from trackers.core.bytetrack.tracker import ByteTrackTracker
    from trackers.core.deepsort.feature_extractor import DeepSORTFeatureExtractor
    from trackers.core.deepsort.tracker import DeepSORTTracker

    __all__.extend(["DeepSORTFeatureExtractor", "DeepSORTTracker"])
except ImportError:
    logger.warning(
        "DeepSORT dependencies not installed. DeepSORT features will not be available. "
        "Please run `pip install trackers[deepsort]` and try again."
    )
    pass
try:
    from trackers.core.bytetrack.tracker import ByteTrackTracker

    __all__.extend(["ByteTrackTracker", "DeepSORTFeatureExtractor", "DeepSORTTracker"])
except ImportError:
    logger.warning(
        "ByteTrack dependencies not installed. ByteTrack features will not be available. "
        "Please run `pip install trackers` and try again."
    )
