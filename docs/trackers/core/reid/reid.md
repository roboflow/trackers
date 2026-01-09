---
comments: true
---

# Re-Identification (ReID)

Re-identification (ReID) enables object tracking systems to recognize the same object or identity across different frames—even when occlusion, appearance changes, or re-entries occur. This is essential for robust, long-term multi-object tracking.

## Installation

To use ReID features in the trackers library, install the package with the appropriate dependencies for your hardware:

!!! example "Install trackers with ReID support"

    === "CPU"
        ```bash
        pip install "trackers[reid,cpu]"
        ```

    === "CUDA 11.8"
        ```bash
        pip install "trackers[reid,cu118]"
        ```

    === "CUDA 12.4"
        ```bash
        pip install "trackers[reid,cu124]"
        ```

    === "CUDA 12.6"
        ```bash
        pip install "trackers[reid,cu126]"
        ```

    === "ROCm 6.1"
        ```bash
        pip install "trackers[reid,rocm61]"
        ```

    === "ROCm 6.2.4"
        ```bash
        pip install "trackers[reid,rocm624]"
        ```

## ReIDModel

The `ReIDModel` class provides a flexible interface to extract appearance features from object detections, which can be used by trackers to associate identities across frames.

### Loading a ReIDModel

You can initialize a `ReIDModel` from any supported pretrained model in the [`timm`](https://huggingface.co/docs/timm/en/index) library using the `from_timm` method.

```python
from trackers import ReIDModel

reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
```

### Supported Models

The `ReIDModel` supports all models available in the timm library. You can list available models using:

```python
import timm
print(timm.list_models())
```

### Extracting Embeddings

To extract embeddings (feature vectors) from detected objects in an image frame, use the `extract_features` method. It crops each detected bounding box from the frame, applies necessary transforms, and passes the crops through the backbone model:

```python
import cv2
import supervision as sv
from trackers import ReIDModel
from inference import get_model

reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
model = get_model(model_id="yolov11m-640")

image = cv2.imread("<INPUT_IMAGE_PATH>")

result = model.infer(image)[0]
detections = sv.Detections.from_inference(result)
features = reid_model.extract_features(image, detections)
```

## Tracking Integration

ReID models are integrated into trackers like DeepSORT to improve identity association by providing appearance features alongside motion cues.

```python
import supervision as sv
from trackers import DeepSORTTracker, ReIDModel
from inference import get_model

reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
tracker = DeepSORTTracker(reid_model=reid_model)
model = get_model("rfdetr-base")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    detections = tracker.update(detections, frame)
    return annotator.annotate(frame, detections, labels=detections.tracker_id)

sv.process_video(
    source_path="<INPUT_VIDEO_PATH>",
    target_path="<OUTPUT_VIDEO_PATH>",
    callback=callback,
)
```

This setup extracts appearance embeddings for detected objects and uses them in the tracker to maintain consistent IDs across frames.

## Training

You can train a custom ReID model using the `TripletsDataset` class, which provides triplets of anchor, positive, and negative samples for metric learning.

Fine-tuning a pre-trained ReID model or training one from scratch can be beneficial when:

- Your target domain (specific camera angles, lighting, object appearances) differs significantly from the data the pre-trained model was exposed to.

- You have a custom dataset featuring unique identities or appearance variations not covered by generic models.

- You aim to boost performance for specific tracking scenarios where general models might underperform. This allows the model to learn features more specific to your data.

### Dataset Structure

Prepare your dataset with the following directory structure, where each subfolder represents a unique identity:

```rext
root/
├── identity_1/
│   ├── image_1.png
│   ├── image_2.png
│   └── image_3.png
├── identity_2/
│   ├── image_1.png
│   ├── image_2.png
│   └── image_3.png
├── identity_3/
│   ├── image_1.png
│   ├── image_2.png
│   └── image_3.png
...
```

Each folder contains images of the same object or person under different conditions.

```python
from torch.utils.data import DataLoader
from trackers.core.reid.dataset.base import TripletsDataset
from trackers import ReIDModel

train_dataset = TripletsDataset.from_image_directories(
    root_directory="<DATASET_ROOT_DIRECTORY>",
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")

reid_model.train(
    train_loader,
    epochs=10,
    projection_dimension=len(train_dataset),
    freeze_backbone=True,
    learning_rate=5e-4,
    weight_decay=1e-2,
    checkpoint_interval=5,
)
```

## Metrics and Monitoring

During training, the model monitors metrics such as triplet loss and triplet accuracy to evaluate embedding quality.

- Triplet Loss: Encourages embeddings of the same identity to be close and different identities to be far apart.

- Triplet Accuracy: Measures how often the model correctly ranks positive samples closer than negatives.

You can enable logging to various backends (matplotlib, TensorBoard, Weights & Biases) during training for real-time monitoring:

```python
from torch.utils.data import DataLoader
from trackers.core.reid.dataset.base import TripletsDataset
from trackers import ReIDModel

train_dataset = TripletsDataset.from_image_directories(
    root_directory="<DATASET_ROOT_DIRECTORY>",
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")

reid_model.train(
    train_loader,
    epochs=10,
    projection_dimension=len(train_dataset),
    freeze_backbone=True,
    learning_rate=5e-4,
    weight_decay=1e-2,
    checkpoint_interval=5,
    log_to_matplotlib=True,
    log_to_tensorboard=True,
    log_to_wandb=True,
)
```

To use the logging capabilities for Matplotlib, TensorBoard, or Weights & Biases, you might need to install additional dependencies.

```bash
pip install "trackers[metrics]"
```

## Resuming from Checkpoints

You can load custom-trained weights or resume training from a checkpoint:

```python
from trackers import ReIDModel

reid_model = ReIDModel.from_timm("<PATH_TO_CUSTOM_SAFETENSORS_CHECKPOINT>")
```

## API


::: trackers.core.reid.model.ReIDModel

::: trackers.core.reid.dataset.base.TripletsDataset
