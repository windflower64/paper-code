# Data and supervision

Create the following layout separately for AntiUAV and UAV-CB. Paths are passed through `--data-root`; no workstation drive letters are required.

```text
dataset/
  annotations/{train,val,test}.json
  train/visible/images/...
  train/thermal/images/...
  train/thermal/labels/...
  val/visible/images/...
  val/thermal/images/...
  val/thermal/labels/...
  test/visible/images/...
  test/thermal/images/...
  test/thermal/labels/...
  sam_masks/train/records.json
  sam_masks/train/masks/000001.png
```

Visible annotations use COCO detection format with a single `UAV` category whose ID is **0**. `images[].width,height` describe the original visible image. Thermal filenames match visible filenames. Thermal label files are YOLO format `0 cx cy width height` in normalized **thermal** coordinates. Empty files represent annotated negative images; missing files are treated as an error by the preparation tool. Never replace thermal labels with visible boxes.

`tools/prepare_data.py` checks train/val/test filename disjointness and paired files, generates `ir_train.json`, `ir_val.json`, `ir_test.json` using original thermal dimensions, and fits a fixed affine calibration using unambiguous **training** annotation pairs only. The calibration is shared by the two fusion levels. Run it before training or adapting the method to a new dataset.

## Offline SAM supervision

The detector does not run SAM during training or inference. Generate RGB training masks offline with SAM3 using the `drone` text prompt and annotated boxes, and retain accepted masks together with their quality records. SAM model installation and its licensed weights are separate dependencies. Alternatively, obtain the authors' precomputed training-mask release asset when available. This repository does not substitute box rectangles for SAM supervision.

Each PNG is a binary instance mask at the original RGB image resolution; nonzero pixels indicate foreground. Provide one record for every training image ID, including rejected masks. The supplied mask generator applies the accepted bbox/jitter screening protocol and writes an explicit binary supervision_weight. Example of its record format:

```json
[
  {"image_id": 1, "file_name": "image_001.jpg", "accepted": true,
   "supervision_weight": 1.0, "mask_bbox_iou": 0.92, "prompt_mask_stability": 0.97},
  {"image_id": 2, "file_name": "image_002.jpg", "accepted": false}
]
```

For accepted single-target records, the default filename is the zero-padded COCO image ID, e.g. `masks/000001.png`. Preserve the supplied explicit supervision_weight. The loader also supports older records using predicted-IoU/stability quality fields, but those are a different weighting format; do not invent such scores or substitute them for the generator's explicit acceptance weight. Rejected masks leave detection supervision intact.

The released accepted checkpoints use the original single-target SBR loss. Its behavior on multi-instance training images has not been established by this public smoke test. Keep the dataset protocol matching the associated checkpoint when reproducing it; evaluating an extended multi-target variant requires separate validation.

No validation or test masks are needed or loaded. Ground-truth masks do not enter inference.

## Split policy

Public training selects checkpoints on val and evaluates test only through `evaluate.py`. The historical supplied weights retain their original development-selection metadata. Using a distinct public val split changes checkpoint selection and may change the reported metrics; this is not an exact replay of historical test-selected experiments.
