"""Evaluate a public-format checkpoint on an explicit split (default: test)."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import torch
torch.set_num_threads(4)
from scf import training as t
from src.core import YAMLConfig
from scf.model import Detector

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--calibration',required=True);p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--confidence',type=float,default=.25);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cuda')
    p.add_argument('--output',default='outputs/evaluation.json');p.add_argument('--max-batches',type=int);a=p.parse_args()
    checkpoint=torch.load(a.checkpoint,map_location='cpu')
    stage=checkpoint.get('metadata',{}).get('stage',checkpoint.get('protocol',{}).get('stage','JOINT'))
    del checkpoint
    cfg=t.configuration(a.config,a.data_root,'IR' if stage=='IR' else 'FUSION',a.batch_size,0,False,False)
    d=cfg.yaml_cfg['val_dataloader']['dataset'];root=Path(a.data_root)
    if stage=='IR':
        d.update(img_folder=str(root/a.split/'thermal/images'),ann_file=str(root/'annotations'/f'ir_{a.split}.json'))
    else:
        d.update(img_folder=str(root/a.split/'visible/images'),ann_file=str(root/'annotations'/f'{a.split}.json'),
            infrared_folder=str(root/a.split/'thermal/images'),infrared_label_folder=str(root/a.split/'thermal/labels'))
    model=Detector(a.config,a.checkpoint,a.calibration).to(a.device)
    stage=model.metadata.get('stage','JOINT')
    metrics,_=t.evaluate(cfg,model.base,model.refiner,model.s16,stage,a.device,a.confidence,a.max_batches)
    metrics.update(split=a.split,checkpoint=str(Path(a.checkpoint).resolve()),partial=bool(a.max_batches))
    t.save_json(a.output,metrics)

if __name__=='__main__':main()
