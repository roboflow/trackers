import supervision as sv
from rfdetr import RFDETRBase

from trackers import ByteTrackTracker
from trackers.core.deepsort.feature_extractor import DeepSORTFeatureExtractor

model = RFDETRBase(device="cuda")  # Load the Object Detector
feature_extractor = DeepSORTFeatureExtractor.from_timm(
    model_name="mobilenetv4_conv_small.e1200_r224_in1k",
)
# Find more info in: https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k
tracker = ByteTrackTracker(
    minimum_consecutive_frames=3,
    feature_extractor=feature_extractor,
    lost_track_buffer=100,
)

color = sv.ColorPalette.from_hex(
    [
        "#ffff00",
        "#ff9b00",
        "#ff8080",
        "#ff66b2",
        "#ff66ff",
        "#b266ff",
        "#9999ff",
        "#3399ff",
        "#66ffff",
        "#33ff99",
        "#66ff66",
        "#99ff00",
    ]
)

box_annotator = sv.BoxAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK)

label_annotator = sv.LabelAnnotator(
    color=color,
    color_lookup=sv.ColorLookup.TRACK,
    text_color=sv.Color.BLACK,
    text_scale=1.5,
)


def callback(frame, _):
    # Obtain bounding box predictions from RF-DETR
    detections = model.predict(frame, threshold=0.5)

    # Update tracker with new detections and retrieve updated IDs
    detections = tracker.update(detections, frame)
    # Filter out detections with IDs of -1 (fresh tracks not yet confirmed)
    detections = detections[detections.tracker_id != -1]

    annotated_image = frame.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(
        annotated_image, detections, detections.tracker_id
    )

    return annotated_image


SOURCE_VIDEO_PATH = (
    "<path-to-your-video>.mp4"  # eg: SOURCE_VIDEO_PATH = "traffic_video_1.mp4"
)

TARGET_VIDEO_PATH = "result_video.mp4"
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback,
    show_progress=True,
)
