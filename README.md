# paper-code

Implementation of RGB–thermal UAV detection with learnable feature sampling, cross-modal interaction, foreground–background grouping, and shape-guided bounding-box refinement.

## Requirements

- Python 3.10 or 3.11
- PyTorch 2.3.1 and torchvision 0.18.1
- A compatible CUDA environment for training

Install the detector dependencies and download the initialization checkpoint:

    pip install -r requirements.txt
    python tools/download_pretrained.py

SAM3 is used to generate training supervision offline. Its environment and checkpoint are installed separately; detector inference does not require SAM3.

## Dataset preparation

Organize each dataset according to [DATA.md](docs/DATA.md). Visible images use COCO annotations, and thermal images use their own annotations. The preparation utility validates the split structure, generates thermal COCO annotations, and estimates the spatial calibration from training pairs.

    python tools/generate_masks.py --data-root data/antiuav --checkpoint /path/to/sam3.pt
    python tools/prepare_data.py --data-root data/antiuav --calibration-output configs/antiuav_calibration.json

Run mask generation in the SAM3 environment and data preparation in the detector environment. The public training entry uses train for optimization, val for checkpoint selection, and test for final evaluation.

## Training

The complete pipeline comprises RGB and thermal initialization, feature-fusion training, boundary refinement, and joint fine-tuning.

AntiUAV:

    python train.py --config configs/antiuav.yml --data-root data/antiuav --calibration configs/antiuav_calibration.json --official weights/dfine_n_coco.pth --stage ALL --output outputs/antiuav

UAV-CB:

    python train.py --config configs/uavcb.yml --data-root data/uavcb --calibration configs/uavcb_calibration.json --official weights/dfine_n_coco.pth --stage ALL --output outputs/uavcb

Append --resume to restore an interrupted run from its last completed epoch. Stage-specific commands, optimization settings, and diagnostic options are provided in [TRAINING.md](docs/TRAINING.md).

## Pretrained checkpoints

Download the checkpoints from [Release v1.0.0](https://github.com/windflower64/paper-code/releases/tag/v1.0.0) and place them in the weights directory.

| Dataset | Checkpoint |
|---|---|
| AntiUAV | [Download](https://github.com/windflower64/paper-code/releases/download/v1.0.0/scf_det_antiuav.pth) |
| UAV-CB | [Download](https://github.com/windflower64/paper-code/releases/download/v1.0.0/scf_det_uavcb.pth) |

Checkpoint-specific metrics and selection details are provided in the accompanying metadata files. SHA-256 checksums are recorded in release_manifest.json.

    python tools/check_weights.py

## Evaluation

    python evaluate.py --config configs/antiuav.yml --data-root data/antiuav --calibration configs/antiuav_calibration.json --checkpoint weights/scf_det_antiuav.pth --split test --confidence 0.25 --output outputs/antiuav_test.json

For UAV-CB, use its configuration, dataset root, calibration, and checkpoint. Evaluation reports COCO AP and fixed-threshold precision, recall, and F1. The matching IoU for these three metrics is 0.5.

## Inference

    from scf.model import Detector

    model = Detector(
        "configs/antiuav.yml",
        "weights/scf_det_antiuav.pth",
        "configs/antiuav_calibration.json",
    ).cuda().eval()
    predictions = model(images, original_sizes)

The input tensor has shape [B, 6, 512, 640], with RGB channels followed by three-channel thermal input. Pixel values are scaled to [0, 1]. Resize the native modalities independently. Original visible-image sizes are supplied in [width, height] order.

## Repository structure

| Path | Description |
|---|---|
| train.py | Staged training and checkpoint restoration |
| evaluate.py | Dataset evaluation |
| scf/ | Model assembly, fusion, and internal attention implementation |
| src/ | Detector, grouping supervision, and boundary refinement |
| configs/ | Dataset and model configurations |
| tools/ | Data preparation, offline supervision, and weight utilities |
| docs/ | Data format, training instructions, and validation record |
| weights/ | Checkpoint metadata |

## License

License texts are provided in LICENSE, LICENSE.DFINE, and LICENSE.ATTENTION. Third-party attribution is recorded in [NOTICE.md](NOTICE.md).
