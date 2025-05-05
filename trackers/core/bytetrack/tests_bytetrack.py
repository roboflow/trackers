import numpy as np
import pytest
import supervision as sv
from trackers import  ByteTrackTracker
from unittest.mock import MagicMock
from trackers.core.bytetrack.kalman_box_tracker import ByteTrackKalmanBoxTracker
import numpy as np
#Run with Pytest : pytest tests_bytetrack.py -v
class DummyTracker(ByteTrackKalmanBoxTracker):
    """A dummy tracker that stores features and returns pre-set features."""
    def __init__(self, bbox, feature=None):
        super().__init__(bbox=bbox, feature=feature)
        self.updated = False

    @staticmethod
    def get_next_tracker_id():
        # Simplify sequential id generation
        ByteTrackKalmanBoxTracker.count_id += 1
        return ByteTrackKalmanBoxTracker.count_id


@pytest.fixture(autouse=True)
def reset_tracker_id():
    # Reset global tracker id before each test
    ByteTrackKalmanBoxTracker.count_id = 0
    yield
    ByteTrackKalmanBoxTracker.count_id = 0

@ pytest.fixture
def tracker():
    return ByteTrackTracker(lost_track_buffer=10, frame_rate=10.0,
                            track_activation_threshold=0.5,
                            minimum_consecutive_frames=0,
                            minimum_iou_threshold=0.1,
                            high_prob_boxes_threshold=0.5)

@ pytest.fixture
def sample_frame():
    # Dummy frame used for feature extraction
    return np.zeros((100, 100, 3), dtype=np.uint8)

@ pytest.fixture
def detections():
    # Create 3 sample detections with bounding boxes and confidences
    xyxy = np.array([[0,0,10,10], [10,10,20,20], [20,20,30,30]], dtype=float)
    confidence = np.array([0.6, 0.4, 0.9])
    return sv.Detections(xyxy=xyxy, confidence=confidence)


def test_get_high_and_low_probability_detections(tracker, detections):
    high, low = tracker._get_high_and_low_probability_detections(detections)
    # Indices with confidence >= 0.5: 0 and 2
    assert len(high) == 2
    assert np.all(high.confidence == np.array([0.6, 0.9]))
    # Indices < 0.5: index 1
    assert len(low) == 1
    assert low.confidence[0] == 0.4


def test_get_associated_indices_empty():
    # No trackers or detections
    sim = np.zeros((0,0))
    tracker_list = []
    matched, ut, ud = ByteTrackTracker()._get_associated_indices(
        sim, detection_boxes=np.zeros((0,4)), trackers=tracker_list, min_similarity_thresh=0.0)
    assert matched == []
    assert ut == set()
    assert ud == set()


def test_get_associated_indices_basic():
    # Create a simple matrix of shape (2 trackers, 3 detections)
    sim = np.array([[0.9, 0.1, 0.2], [0.5, 0.8, 0.3]])
    trackers = [None, None]  # placeholder
    # threshold 0.5 should match (0,0) and (1,1)
    detection_boxes = [None, None, None]  # placeholder
    matched, ut, ud = ByteTrackTracker()._get_associated_indices(sim, detection_boxes=detection_boxes,
                                                                 trackers=trackers,
                                                                 min_similarity_thresh=0.5)
    
    assert set(matched) == {(0,0), (1,1)}
    assert ut == set()
    assert ud == {2}


def test_get_appearance_distance_matrix_empty(tracker):
    # Empty trackers or features
    dist = tracker._get_appearance_distance_matrix(np.empty((0,128)), [])
    assert dist.shape == (0,0)
    dist2 = tracker._get_appearance_distance_matrix(np.empty((5,128)), [])
    assert dist2.shape == (0,5)


def test_get_appearance_distance_matrix_basic(tracker):
    # Two trackers and two detection features
    f1 = np.ones(128)
    f2 = 3*np.ones(128)
    # Monkey-patch trackers
    t1 = DummyTracker(bbox=[0,0,1,1], feature=f1)
    t2 = DummyTracker(bbox=[0,0,1,1], feature=f2)
    feats = np.vstack([f2, f1])  # so distances are [1, sqrt(2)], etc.
    dist = tracker._get_appearance_distance_matrix(feats, [t1, t2])
    # distance_matrix[i,j] = distance between tracker j and feature i
    assert dist.shape == (2,2)
    # Check values clipped between 0 and 1
    assert np.all(dist >= 0)
    assert np.all(dist <= 1)


def test_reset(tracker):
    tracker.trackers = [DummyTracker([0,0,1,1], feature=None)]
    tracker.reset()
    assert tracker.trackers == []
    assert ByteTrackKalmanBoxTracker.count_id == 0


def make_detections(bboxes, confidences):
    return sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.zeros(len(bboxes), dtype=int)  # optional
    )
class MockExtractor():
    def __init__(self,features):
        self.features=  features
    def extract_features (self,frame, high_prob_detections): return self.features

def test_update_full_workflow():
    # Mock image (used by feature extractor)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # High-confidence detections: should be matched or spawn trackers
    high_conf_bboxes = [
        [100, 100, 150, 150],  # Detection 1
        [200, 200, 250, 250],  # Detection 2
        [1000, 2000, 2500, 2500] #Detection 3
    ]
    high_confidences = [0.9, 0.95,0.5]
    # Mock the feature extractor if used
    f1 =  [0,1]
    f2 = [1,0]
    f3 = [2,1]
    feature_extractor = MockExtractor([f1,f2,f3])
    # Low-confidence detection: may or may not be matched
    low_conf_bboxes = [
        [300, 300, 350, 350],  # Detection 4
    ]
    low_confidences = [0.4]

    all_bboxes = high_conf_bboxes + low_conf_bboxes
    all_confidences = high_confidences + low_confidences

    detections = make_detections(all_bboxes, all_confidences)

    # Set up the tracker with a dummy feature extractor
    tracker = ByteTrackTracker(
        high_prob_boxes_threshold=0.5,
        distance_metric='cosine',
        feature_extractor=feature_extractor,
        minimum_consecutive_frames = 1,

    )

    # Run first update and all trackers should be set to -1
    output_detections = tracker.update(detections, frame)
    assert (output_detections.tracker_id == -1).all()


    # Run 2nd update in order to start to detect.

    output_detections = tracker.update(detections, frame)
    print(detections.xyxy)
    print(output_detections.tracker_id)

    # --- Some Assertions on expected types ---
    assert isinstance(output_detections, sv.Detections)
    assert hasattr(output_detections, "tracker_id")
    high_conf_mask = output_detections.confidence >= 0.5
    # Check that high-confidence detections have been assigned to its tracker_ids

    assert np.all(output_detections[output_detections.tracker_id == 0].xyxy[0] == detections.xyxy[0])
    assert np.all(output_detections[output_detections.tracker_id == 1].xyxy[0] == detections.xyxy[1])

    # Check that low-confidence detection has tracker_id -1 if unmatched
    low_conf_mask = output_detections.confidence < 0.5
    assert (output_detections.tracker_id[low_conf_mask] == -1).all()

    # After first update, number of internal trackers should be at least 2
    assert len(tracker.trackers) >= 2


    # Now change order of features 
    feature_extractor_inverted = MockExtractor([f2,f1,f3])
    
    low_conf_bboxes = [
        [1000, 2000, 2500, 2500],  # Detection 3 reappears as low confidence and should match that tracker
    ]
    low_confidences = [0.4]

    all_bboxes = high_conf_bboxes + low_conf_bboxes
    all_confidences = high_confidences + low_confidences

    detections = make_detections(all_bboxes, all_confidences)
    tracker.feature_extractor = feature_extractor_inverted
    output_detections = tracker.update(detections, frame)

    assert np.all(output_detections[output_detections.tracker_id == 1].xyxy[0] == detections.xyxy[0])# Associates correctly feature changing order of appearance
    assert np.all(output_detections[output_detections.tracker_id == 0].xyxy[0] == detections.xyxy[1]) # Associates correctly feature changing order of appearance
    assert np.all(output_detections[output_detections.tracker_id == 2].xyxy[0] == detections.xyxy[-1]) #Associates correctly low confidence object that was high confidence

    print("Test passed: Full ByteTrack-style update works.")