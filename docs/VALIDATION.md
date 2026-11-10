# Public-entry validation

Checked locally on 2026-11-09 with the existing detector environment (PyTorch 2.3.1/CUDA, RTX 4080).

Passed:

- Original accepted AntiUAV and UAV-CB weights: strict parameter loading.
- Data preparation: paired-file/split checks, separate IR COCO annotation generation and train-only affine fitting on a small real-image fixture.
- RGB and IR stages: real data, forward/backward, finite gradients, optimizer step, val evaluation and checkpoints.
- Complete-model FUSION: detector and learnable fusion training with positive SGC loss.
- REFINE: frozen detector, SBR/occupancy loss and refiner EMA.
- JOINT: detector/SBR optimization and context-disabled evaluation.
- Serial ALL command: all five stages completed with the public UAV-CB configuration on the real-image diagnostic fixture.
- Epoch-boundary pause/resume: epoch 0 saved, optimizer/model restored, epoch 1 trained and logged.
- Separate evaluation command: a newly trained full-model checkpoint loaded and evaluated on an explicitly selected split.
- Source files compile; Git exclusions protect datasets, masks, checkpoints, caches and private research snapshots.
- Fresh GitHub clone: the five-stage ALL smoke pipeline also completes using only the cloned source and externally supplied diagnostic data/initialization weights.

The fixture contains four train images and disjoint val/test images from AntiUAV. Using the UAV-CB configuration in the serial check verifies configuration/architecture compatibility, not UAV-CB benchmark accuracy. Short-run checkpoints and metrics are marked smoke. A full clean-environment train/val-selected benchmark run has not been performed here.

The SAM3 generator follows the existing local teacher-generation implementation; fresh SAM3 model inference was not part of these detector checks. Install its official optional environment and checkpoint separately, or use matching precomputed training masks.
