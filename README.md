# paper-code — SCF-Det

RGB–thermal UAV detection using learnable thermal sampling, pre-encoder cross-modal interaction, SAM-guided grouping, and shape-guided box refinement.

## Release status

This release includes the complete-model serial training pipeline, initialization stages, offline mask generation, evaluation, and two accepted checkpoints. Short real-data tests cover each stage, checkpoint loading and epoch-boundary resume. See [TRAINING.md](docs/TRAINING.md) for the executable workflow. Full-run metrics under the public val-selection protocol have not been measured.

The public configuration uses **train for fitting, val for checkpoint selection, and test for final evaluation**. Files under `configs/` use relative dataset paths. Existing checkpoints were selected using the original test split as a development set; their metadata retains that provenance. They must not be described as val-selected checkpoints. New val-selected checkpoints require training under the public protocol.

## Contents

- `src/`: D-FINE-based detector, SGC and SBR implementation.
- `scf/`: portable current-fusion assembly and inference model wrapper.
- `train.py`: full-model serial training, individual stages, and resume (no component-ablation interface).
- `evaluate.py`: explicit val/test evaluation and P/R/F1 recount.
- `tools/generate_masks.py` and `tools/prepare_data.py`: offline SAM3 teachers, paired-data checks, thermal annotations and train-only calibration.
- [Validation record](docs/VALIDATION.md): checks completed and their scope.
- `third_party/icafusion/`: the author interaction blocks used by the fusion adapter, with license.
- `configs/`: resolved dataset configurations and train-derived spatial calibration.
- `weights/`: inference state dictionaries, metrics and SHA-256 provenance for AntiUAV and UAV-CB.
- `research_snapshot/`: local-only audit archive of training entry points and dependencies. It contains historical absolute paths and is excluded from Git. These files are not portable public training commands.
- `release_manifest.json`: exported checkpoint inventory and release status.

## Checkpoints

Upload the `.pth` files as **GitHub Release assets**, not ordinary Git objects. Their metadata and checksums can remain in Git. Optimizer states and redundant epoch checkpoints are excluded from the export. Original resumable training checkpoints remain in the research workspace.

The AntiUAV checkpoint is the accepted new-fusion JOINT checkpoint. The UAV-CB checkpoint is the accepted REFINE checkpoint. The separate new ablation queue is not silently substituted for these accepted models.

## Inference API and compatibility check

Run from the repository root after installing the dependencies:

```bash
python tools/check_weights.py
```

```python
from scf.model import SCFDetector
model = SCFDetector('configs/antiuav.yml', 'weights/scf_det_antiuav.pth',
                    'configs/antiuav_calibration.json').cuda().eval()
# images: [B,6,512,640], RGB followed by IR replicated to three channels,
# float values in [0,1]; resize the native modalities independently.
# original_sizes: [B,2], original RGB [width,height], on the same device.
predictions = model(images, original_sizes)
```

The compatibility check verifies parameter keys and shapes, not AP. The weights retain their associated measured metrics; short public-entry tests are not new benchmark measurements.

## Attribution and licensing

The detector derives from [D-FINE](https://github.com/Peterande/D-FINE), licensed under Apache-2.0 (`LICENSE.DFINE`). The cross-modal interaction block derives from [ICAFusion](https://github.com/chanchanchan97/ICAFusion), licensed under AGPL-3.0. The combined release carries AGPL-3.0 in `LICENSE`; upstream notices are retained. The interaction block is reused and must not be represented as an independently invented attention mechanism.

No dataset images, annotations, SAM masks, credentials or private Git history are included in the public Git file set.

## Before publication

1. Review the public train/val/test protocol and dataset licenses.
2. For val-selected benchmark results, train the documented recipe to completion.
3. Freeze the matching manuscript, architecture diagram and results.
4. Create the GitHub repository named `paper-code`, push reviewed source, and attach weights to a release.
