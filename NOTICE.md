# Third-party notices

## Detector foundation

The D-FINE-derived files under src/ retain their upstream copyright notices. Source: https://github.com/Peterande/D-FINE. License: Apache-2.0, supplied in LICENSE.DFINE.

## Attention implementation

The classes in scf/_attention.py were extracted from ICAFusion's models/common.py and integrated into the internal fusion implementation. Source: https://github.com/chanchanchan97/ICAFusion. Authors: Jifeng Shen, Yifei Chen, Yue Liu, Xin Zuo, Heng Fan, and Wankou Yang. License: AGPL-3.0, supplied in LICENSE.ATTENTION and LICENSE. The extraction header and upstream implementation are retained.

Repository integration changes include the module location and import paths. The attention mathematics and pretrained parameter layout are unchanged by this reorganization.
