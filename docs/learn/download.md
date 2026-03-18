# Download Datasets

Download benchmark multi-object tracking datasets for evaluation and development. Supported datasets can be downloaded in full or filtered by split and asset type.

**What you'll learn:**

- Download MOT17 and SportsMOT benchmark datasets
- Select specific splits and asset types
- Use the download cache to avoid re-downloading

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/SportsMOT_v_-6Os86HzwCs_c001-1280x720.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>Visualization of ground-truth annotations for SportsMOT.</small></p>

---

## Install

Get started by installing the package.

```text
pip install trackers
```

For more options, see the [install guide](install.md).

---

## Available Datasets

The table below lists every dataset you can download, along with its splits, assets, and license. Assets include raw video frames, ground-truth bounding box annotations, and pre-computed detections.

|   Dataset    |                               Description                               |         Splits         |                Assets                 |     License     |
| :----------: | :---------------------------------------------------------------------: | :--------------------: | :-----------------------------------: | :-------------: |
|   `mot17`    |    Pedestrian tracking with crowded scenes and frequent occlusions.     | `train`, `val`, `test` | `frames`, `annotations`, `detections` | CC BY-NC-SA 3.0 |
| `sportsmot`  | Sports broadcast tracking with fast motion and similar-looking targets. | `train`, `val`, `test` |        `frames`, `annotations`        |    CC BY 4.0    |
| `dancetrack` |                             *Coming soon.*                              |           —            |                   —                   |        —        |
| `soccernet`  |                             *Coming soon.*                              |           —            |                   —                   |        —        |

=== "CLI"

    Use `--list` to print available datasets, splits, and asset types.

    ```text
    trackers download --list
    ```

=== "Python"

    Iterate over the `Dataset` enum to list supported datasets.

    ```python
    from trackers import Dataset

    for dataset in Dataset:
        print(dataset.value)
    ```

---

## Quickstart

Pass a dataset name to download all of its splits and assets.

=== "CLI"

    Download the full MOT17 dataset.

    ```text
    trackers download mot17
    ```

=== "Python"

    Download the full MOT17 dataset.

    ```python
    from trackers import Dataset, download_dataset

    download_dataset(dataset=Dataset.MOT17)
    ```

---

## Selective Downloads

Full datasets can be large. Narrow your download to specific splits and asset types.

=== "CLI"

    Use `--split` and `--asset` to filter by split, asset type, or both.

    ```text
    trackers download mot17 --split train --asset annotations
    ```

    ```text
    trackers download mot17 --split train,val --asset annotations,frames
    ```

    ```text
    trackers download sportsmot --split val --asset annotations
    ```

=== "Python"

    Specify splits and assets as enums, strings, or lists of either.

    ```python
    from trackers import Dataset, DatasetAsset, DatasetSplit, download_dataset

    download_dataset(
        dataset=Dataset.MOT17,
        split=DatasetSplit.TRAIN,
        asset=DatasetAsset.ANNOTATIONS,
    )
    ```

    ```python
    from trackers import Dataset, DatasetAsset, DatasetSplit, download_dataset

    download_dataset(
        dataset=Dataset.MOT17,
        split=[DatasetSplit.TRAIN, DatasetSplit.VAL],
        asset=[DatasetAsset.ANNOTATIONS, DatasetAsset.FRAMES],
    )
    ```

    ```python
    from trackers import Dataset, DatasetAsset, DatasetSplit, download_dataset

    download_dataset(
        dataset=Dataset.SPORTSMOT,
        split=DatasetSplit.VAL,
        asset=DatasetAsset.ANNOTATIONS,
    )
    ```

---

## Output

Dataset files are extracted to the current directory by default. Set a custom output path to change this.

=== "CLI"

    Use `--output` to extract into a custom directory.

    ```text
    trackers download mot17 \
        --split train,val \
        --asset annotations,frames \
        --output ./datasets
    ```

=== "Python"

    Use `output` to extract into a custom directory.

    ```python
    from trackers import Dataset, DatasetAsset, DatasetSplit, download_dataset

    download_dataset(
        dataset=Dataset.MOT17,
        split=[DatasetSplit.TRAIN, DatasetSplit.VAL],
        asset=[DatasetAsset.ANNOTATIONS, DatasetAsset.FRAMES],
        output="./datasets",
    )
    ```

The resulting directory structure after extraction.

```text
datasets/
└── mot17/
    ├── train/
    │   ├── MOT17-02-FRCNN/
    │   │   ├── gt/
    │   │   │   └── gt.txt
    │   │   └── img1/
    │   │       ├── 000001.jpg
    │   │       ├── 000002.jpg
    │   │       └── ...
    │   ├── MOT17-04-FRCNN/
    │   │   ├── gt/
    │   │   │   └── gt.txt
    │   │   └── img1/
    │   │       └── ...
    │   └── ...
    └── val/
        ├── MOT17-02-FRCNN/
        │   ├── gt/
        │   │   └── gt.txt
        │   └── img1/
        │       └── ...
        └── ...
```

---

## Caching

Every downloaded ZIP is saved to `~/.cache/trackers` and verified with an MD5 checksum. Future runs skip the download and extract directly from the cache.

=== "CLI"

    Use `--cache-dir` to store ZIPs in a custom location.

    ```text
    trackers download mot17 \
        --split train \
        --asset annotations \
        --cache-dir ./my-cache
    ```

=== "Python"

    Use `cache_dir` to store ZIPs in a custom location.

    ```python
    from trackers import Dataset, DatasetAsset, DatasetSplit, download_dataset

    download_dataset(
        dataset=Dataset.MOT17,
        split=DatasetSplit.TRAIN,
        asset=DatasetAsset.ANNOTATIONS,
        cache_dir="./my-cache",
    )
    ```

---

## CLI Reference

All arguments accepted by the `trackers download` command.

<table>
  <colgroup>
    <col style="width: 40%">
    <col style="width: 40%">
    <col style="width: 20%">
  </colgroup>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Description</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>dataset</code></td>
      <td>Dataset name to download. Options: <code>mot17</code>, <code>sportsmot</code>.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--list</code></td>
      <td>List available datasets, splits, and asset types without downloading.</td>
      <td><code>false</code></td>
    </tr>
    <tr>
      <td><code>--split</code></td>
      <td>Comma-separated splits to download. Omit to download all available splits.</td>
      <td>all</td>
    </tr>
    <tr>
      <td><code>--asset</code></td>
      <td>Comma-separated asset types to download: <code>frames</code>, <code>annotations</code>, <code>detections</code>. Omit to download all available assets.</td>
      <td>all</td>
    </tr>
    <tr>
      <td><code>--output</code></td>
      <td>Directory where dataset files are extracted.</td>
      <td><code>.</code></td>
    </tr>
    <tr>
      <td><code>--cache-dir</code></td>
      <td>Directory for caching downloaded ZIP files. Cached files are verified by MD5 and reused across runs.</td>
      <td><code>~/.cache/trackers</code></td>
    </tr>
  </tbody>
</table>
