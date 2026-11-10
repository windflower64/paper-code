"""Train the RGB-thermal detector through the staged serial pipeline."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import torch
torch.set_num_threads(4)
from scf import training as t

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--stage',choices=['ALL',*t.EPOCHS],default='ALL')
    p.add_argument('--official');p.add_argument('--rgb');p.add_argument('--ir');p.add_argument('--parent');p.add_argument('--calibration')
    p.add_argument('--output',default='outputs/scf');p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--accumulation',type=int);p.add_argument('--epochs',type=int)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--resume',action='store_true')
    p.add_argument('--max-train-batches',type=int);p.add_argument('--max-val-batches',type=int)
    p.add_argument('--stop-after-epochs',type=int,help='Gracefully stop after this many additional epochs; keep resumable last')
    p.add_argument('--smoke',action='store_true',help='Short diagnostic run; never benchmark metrics')
    return p

def main(a):
    a.arm='FULL'
    if a.smoke:
        a.epochs=a.epochs or 1;a.max_train_batches=a.max_train_batches or 2;a.max_val_batches=a.max_val_batches or 1
    if a.stage=='ALL':
        if not a.smoke and (a.epochs or a.max_train_batches or a.max_val_batches):raise ValueError('Pipeline diagnostic overrides require --smoke')
        root=Path(a.output)
        for stage in t.EPOCHS:
            if stage in ('RGB','IR') and getattr(a,stage.lower()):continue
            if stage=='IR' and not t.ARMS[a.arm][2]:continue
            if stage=='REFINE' and not t.ARMS[a.arm][1]:continue
            out=root/stage
            if (out/'complete.json').exists():continue
            args=copy.deepcopy(a);args.stage=stage;args.output=str(out)
            args.rgb=a.rgb or str(root/'RGB/best.pth');args.ir=a.ir or str(root/'IR/best.pth')
            if stage in ('REFINE','JOINT'):
                parent='FUSION' if stage=='REFINE' or not t.ARMS[a.arm][1] else 'REFINE'
                args.parent=str(root/parent/'best.pth')
            args.resume=a.resume and (out/'last.pth').exists()
            main(args)
            if not (out/'complete.json').exists():
                print(f'Pipeline paused at {stage}; resume with --resume',flush=True);return
        return
    if not torch.cuda.is_available() and a.device.startswith('cuda'):raise RuntimeError('CUDA unavailable; choose --device cpu for diagnostics')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pth').exists() and not a.resume:raise FileExistsError('Existing run: use --resume or a new --output')
    sgc,sbr,_=t.ARMS[a.arm]
    if a.stage in ('RGB','IR'):sgc=sbr=0
    t.seed(a.seed);torch.backends.cudnn.benchmark=False
    cfg=t.configuration(a.config,a.data_root,a.stage,a.batch_size,a.workers,sgc,sbr)
    train=cfg.train_dataloader.dataset;val=cfg.val_dataloader.dataset
    names=lambda d:{d.coco.imgs[i]['file_name'] for i in d.ids}
    if names(train)&names(val):raise ValueError('Training and validation filenames overlap')
    base,refiner,criterion,s16,provenance=t.build(cfg,a.stage,a.arm,a.device,a.official,a.rgb,a.ir,a.parent,a.calibration)
    opt=t.optimizer(base,refiner,a.stage);epochs=a.epochs or t.EPOCHS[a.stage]
    accumulation=a.accumulation or (2 if a.stage in ('RGB','IR','FUSION') else 1)
    ema=copy.deepcopy(refiner).eval().requires_grad_(False) if a.stage=='REFINE' else None
    protocol=dict(arm=a.arm,stage=a.stage,epochs=epochs,batch_size=a.batch_size,accumulation=accumulation,
        seed=a.seed,selection_split='val',precision='FP32',train_images=len(train),val_images=len(val),
        smoke=bool(a.max_train_batches or a.max_val_batches),config=str(Path(a.config).resolve()),
        data_root=str(Path(a.data_root).resolve()),initialization=provenance,
        sources={k:v for k,v in vars(a).items() if k in ('official','rgb','ir','parent','calibration')})
    start=0;best=-1.
    if a.resume:
        saved=torch.load(out/'last.pth',map_location='cpu')
        if saved['protocol']!=protocol:raise ValueError('Resume configuration differs from saved protocol')
        base.load_state_dict(saved['base']);refiner.load_state_dict(saved['refiner']);opt.load_state_dict(saved['optimizer'])
        if ema is not None:ema.load_state_dict(saved['ema'])
        start=saved['next_epoch'];best=saved['best_ap']
    t.save_json(out/'protocol.json',protocol)
    for epoch in range(start,epochs):
        t.seed(10000+a.seed+epoch);loader=cfg.train_dataloader;loader.set_epoch(epoch)
        base.set_training_epoch(max(epoch-2,0) if a.stage=='FUSION' else epoch);t.modes(base,refiner,a.stage)
        limit=min(len(loader),a.max_train_batches or len(loader));total=0.;tick=time.monotonic();opt.zero_grad(set_to_none=True)
        for i,(samples,targets) in enumerate(loader):
            if i>=limit:break
            scale=t.factor(a.stage,epoch+i/max(len(loader),1),epochs)
            for g in opt.param_groups:g['lr']=g['initial_lr']*scale
            samples,targets=t.move(samples,targets,a.device)
            loss,stats=t.loss_step(base,refiner,criterion,s16,samples,targets,a.stage,epoch)
            if not torch.isfinite(loss):raise FloatingPointError(stats)
            group=min(accumulation,limit-(i//accumulation)*accumulation)
            (loss/group).backward();total+=float(loss.detach())
            if (i+1)%accumulation==0 or i+1==limit:
                params=[p for g in opt.param_groups for p in g['params']]
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
                opt.step();opt.zero_grad(set_to_none=True)
                if ema is not None:
                    with torch.no_grad():
                        for p,q in zip(ema.parameters(),refiner.parameters()):p.lerp_(q,.01)
            if i%10==0:print(json.dumps(dict(stage=a.stage,epoch=epoch,step=i,steps=limit,loss=float(loss.detach()),**stats)),flush=True)
        metrics,predictions=t.evaluate(cfg,base,ema if ema is not None else refiner,s16,a.stage,a.device,max_batches=a.max_val_batches)
        ap=metrics['coco_eval_bbox'][0]
        eligible=a.stage!='FUSION' or epoch>=2 or protocol['smoke']
        improved=eligible and ap>best
        if eligible:best=max(best,ap)
        row=dict(epoch=epoch,loss=total/max(limit,1),validation=metrics,seconds=time.monotonic()-tick)
        if improved:
            t.save(out/'best.pth',dict(base=base.state_dict(),refiner=(ema if ema is not None else refiner).state_dict(),
                protocol=protocol,metadata=dict(stage=a.stage,model_selection_split='val',smoke=protocol['smoke']),epoch=epoch))
            t.save_json(out/'best_metrics.json',row);t.save(out/'best_predictions.pth',predictions)
        t.save(out/'last.pth',dict(base=base.state_dict(),refiner=refiner.state_dict(),optimizer=opt.state_dict(),
            ema=ema.state_dict() if ema is not None else None,protocol=protocol,next_epoch=epoch+1,best_ap=best))
        with (out/'log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if a.stop_after_epochs and epoch+1-start>=a.stop_after_epochs and epoch+1<epochs:
            print('PAUSED_AT_EPOCH_BOUNDARY; resume with --resume',flush=True);return
    t.save_json(out/'complete.json',dict(epochs=epochs,best_val_ap=best,smoke=protocol['smoke']))

if __name__=='__main__':main(parser().parse_args())
