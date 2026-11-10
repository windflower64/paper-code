# Complete-model training

Run all commands from the repository root. This public interface trains the full model; component-ablation experiments are not included.

## 1. Install and obtain initialization

Use Python 3.10/3.11, PyTorch 2.3.1 and torchvision 0.18.1 with the appropriate CUDA build. Then install requirements.txt. The local smoke tests used an RTX 4080; a clean Linux installation has not been tested in this session.

    pip install -r requirements.txt
    python tools/download_pretrained.py
    python tools/check_weights.py

The downloader obtains official D-FINE-N COCO weights and verifies SHA-256. Custom backbone parameters absent from the official model are initialized locally; the loading report records unmatched tensors.

## 2. Prepare paired data and offline masks

See DATA.md for the exact layout, category ID 0, native visible annotations and separate thermal YOLO labels. Generate SAM3 training masks in a separately installed official SAM3 environment with its licensed weights:

    python tools/generate_masks.py --data-root data/antiuav --checkpoint /path/to/sam3.pt

This generator uses the existing drone text/box prompt and jitter acceptance protocol. It reads training annotations only. Alternatively use the matching precomputed mask package if supplied. SAM3 mask inference was not re-run during the detector smoke tests; its optional environment remains separate.

Back in the detector environment:

    python tools/prepare_data.py --data-root data/antiuav --calibration-output configs/antiuav_calibration.json

The preparation tool checks paired files and split disjointness, generates IR COCO annotations, and estimates calibration from unambiguous training annotation pairs. Never substitute RGB boxes for IR labels. Repeat with data/uavcb and configs/uavcb_calibration.json for UAV-CB.

## 3. Train the full pipeline

    python train.py --config configs/antiuav.yml --data-root data/antiuav --calibration configs/antiuav_calibration.json --official weights/dfine_n_coco.pth --stage ALL --output outputs/antiuav --batch-size 16 --workers 0

    python train.py --config configs/uavcb.yml --data-root data/uavcb --calibration configs/uavcb_calibration.json --official weights/dfine_n_coco.pth --stage ALL --output outputs/uavcb --batch-size 16 --workers 0

| Stage | Trainable parts | Default epochs |
|---|---|---:|
| RGB | RGB backbone, encoder, decoder | 20 |
| IR | IR detector with thermal-coordinate annotations | 30 |
| FUSION | RGB detector and fusion, with SGC | 22 |
| REFINE | SBR only; detector frozen; refiner EMA | 30 |
| JOINT | RGB detector and SBR; IR/fusion frozen | 20 |

FUSION uses AdamW with learning rates 1e-4 (RGB backbone) and 2e-4 (other trainable detector/fusion parameters). REFINE uses 1e-4. JOINT uses 1e-5 for the detector and 5e-5 for SBR. Weight decay is 1e-4, gradient clipping is 1, and detector training uses FP32. The default physical batch is 16; accumulation is 2 in RGB/IR/FUSION and 1 in REFINE/JOINT. Change accumulation explicitly when changing batch size if the effective batch should remain fixed.

SGC uses S8, weight 10, with FUSION epoch9–14 decay. Fusion samples/interacts at S16/S32. SBR reads S4/S8. JOINT retains the original context-training branch and disables its score contribution during evaluation. Only best.pth and fully resumable last.pth are retained per stage, plus metrics/predictions/logs.

Dataset-local RGB/IR initialization weights can be passed with --rgb and --ir; ALL skips those stages. Their architecture must match. Historical supplied full-model weights also reflect their recorded dataset-specific initialization histories. The executable public recipe uses val selection, so new training is not guaranteed to produce historical test-selected metrics exactly.

## 4. Explicit stages and resume

    python train.py --config configs/antiuav.yml --data-root data/antiuav --calibration configs/antiuav_calibration.json --stage FUSION --rgb outputs/antiuav/RGB/best.pth --ir outputs/antiuav/IR/best.pth --output outputs/antiuav/FUSION

    python train.py --config configs/antiuav.yml --data-root data/antiuav --calibration configs/antiuav_calibration.json --stage REFINE --parent outputs/antiuav/FUSION/best.pth --output outputs/antiuav/REFINE

    python train.py --config configs/antiuav.yml --data-root data/antiuav --calibration configs/antiuav_calibration.json --stage JOINT --parent outputs/antiuav/REFINE/best.pth --output outputs/antiuav/JOINT

Append --resume to the same command to restore optimizer/model/EMA state at the last completed epoch. ALL --resume skips completed stages and resumes the unfinished stage. --stop-after-epochs N pauses at a saved epoch boundary. Interrupted work after that boundary is replayed; mid-minibatch exact continuation is not claimed.

For installation diagnostics, use an explicit stage with --epochs 1 --batch-size 2 --accumulation 1 --max-train-batches 2 --max-val-batches 1 and a new output directory. Such runs are marked smoke and must not be reported as benchmark results.

To test the full serial pipeline, add --smoke --batch-size 2 --accumulation 1 to the ALL command and use a new output directory. This runs one short diagnostic epoch per stage and explicitly marks all results as smoke.

## 5. Evaluate

    python evaluate.py --config configs/antiuav.yml --data-root data/antiuav --calibration configs/antiuav_calibration.json --checkpoint weights/scf_det_antiuav.pth --split test --confidence 0.25 --output outputs/antiuav_test.json

    python evaluate.py --config configs/uavcb.yml --data-root data/uavcb --calibration configs/uavcb_calibration.json --checkpoint weights/scf_det_uavcb.pth --split test --confidence 0.25 --output outputs/uavcb_test.json

For new training, use the desired stage's best.pth. REFINE and JOINT results are separate. Evaluation reports COCO AP and fixed-confidence P/R/F1 with matching IoU 0.5. The confidence option changes P/R/F1, not the AP calculation. Train selects on val; test is evaluated only by this explicit command.

## Implementation map

- scf/fusion.py: calibrated grid, learned offsets, sampling, and credited ICAFusion adapter.
- scf/training.py: model assembly, stage freezing, losses, optimizer, evaluator.
- src/zoo/dfine/: grouping-loss and side-search implementations.
- scf/model.py: full-model inference and checkpoint-compatible assembly.

The real-data smoke checks cover optimization in all full-model stages, saving/loading, actual pause/resume into a second epoch, and the independent evaluation command. These checks establish execution, not full-training accuracy or cross-platform numerical equivalence.
