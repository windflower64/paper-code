"""CPU-only strict architecture/checkpoint compatibility check."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(4)
from scf.model import Detector

root=Path(__file__).resolve().parents[1]
for dataset in ('antiuav','uavcb'):
    model=Detector(root/'configs'/f'{dataset}.yml',root/'weights'/f'scf_det_{dataset}.pth',root/'configs'/f'{dataset}_calibration.json')
    print(dataset,'strict load PASS',sum(p.numel() for p in model.parameters()),flush=True)
    del model
