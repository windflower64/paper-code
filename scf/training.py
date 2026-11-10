"""Training stages for the RGB-thermal detector."""
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.ops import box_iou
from src.core import YAMLConfig
from src.zoo.dfine.sam_side_search import SAMSideSearch
from src.zoo.dfine.sam_side_search_multiscale import SAMSideSearchContext
from .model import FeatureTap
from .fusion import PreEncoderFusion, Sampler, attach_hooks

ARMS={'FULL':(1,1,1)}
EPOCHS={'RGB':20,'IR':30,'FUSION':22,'REFINE':30,'JOINT':20}

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)

def state(path):
    x=torch.load(path,map_location='cpu')
    if 'base' in x:return x['base']
    if isinstance(x.get('ema'),dict):return x['ema'].get('module',x['ema'])
    return x.get('model',x)

def transfer(module,weights,prefixes,strict=False):
    current=module.state_dict();matched=[];missing=[]
    for k,v in current.items():
        if not k.startswith(prefixes):continue
        key=k.replace('thermal_backbone.','backbone.',1).replace('thermal_encoder.','encoder.',1)
        if key in weights and v.shape==weights[key].shape:current[k]=weights[key];matched.append(k)
        else:missing.append(k)
    if strict and missing:raise ValueError(f'Initialization tensors missing: {missing[:12]}')
    if not matched:raise ValueError('No matching initialization parameters')
    module.load_state_dict(current,strict=True)
    return dict(loaded=len(matched),unmatched=missing)

def configuration(path,data_root,stage,batch,workers,sgc,sbr):
    cfg=YAMLConfig(str(path))
    for name in ('train','val'):
        entry=cfg.yaml_cfg[f'{name}_dataloader'];d=entry['dataset']
        d.update(img_folder=str(Path(data_root)/name/'visible/images'),
            ann_file=str(Path(data_root)/'annotations'/f'{name}.json'),
            infrared_folder=str(Path(data_root)/name/'thermal/images'),
            infrared_label_folder=str(Path(data_root)/name/'thermal/labels'),
            sam_mask_root=str(Path(data_root)/'sam_masks/train') if name=='train' and (sgc or sbr) else None)
        entry.update(total_batch_size=batch,num_workers=workers,persistent_workers=workers>0)
        if workers:entry['prefetch_factor']=2
        else:entry.pop('prefetch_factor',None)
        ops=([dict(type='RandomPhotometricDistort',p=.5),dict(type='RandomHorizontalFlip')]
             if name=='train' and stage in ('RGB','IR','FUSION') else [])
        ops.extend([dict(type='Resize',size=[512,640]),dict(type='SanitizeBoundingBoxes',min_size=1),
            dict(type='ConvertPILImage',dtype='float32',scale=True),dict(type='ConvertBoxes',fmt='cxcywh',normalize=True)])
        d['transforms'].update(ops=ops,policy=None)
        if stage=='IR':
            d.update(type='CocoDetection',img_folder=str(Path(data_root)/name/'thermal/images'),
                ann_file=str(Path(data_root)/'annotations'/f'ir_{name}.json'),sam_mask_root=None)
            d.pop('infrared_folder',None);d.pop('infrared_label_folder',None)
    if sgc or sbr:
        if not (Path(data_root)/'sam_masks/train/records.json').is_file():
            raise FileNotFoundError('SAM supervision required: prepare sam_masks/train/records.json and masks/*.png')
    cfg.yaml_cfg['DFINECriterion']['sqer_shape_weight']=0.
    return cfg

def build(cfg,stage,arm,device,official=None,rgb=None,ir=None,parent=None,calibration=None):
    sgc,sbr,fusion=ARMS[arm]
    if stage in ('RGB','IR'):sgc=sbr=fusion=0
    base=cfg.model;reader=base.sqer;base.sqer=FeatureTap()
    refiner=SAMSideSearch(reader) if sbr else nn.Module()
    if fusion:
        if calibration is None:raise ValueError('--calibration required for multimodal fusion')
        base.mote_fusion=PreEncoderFusion(list(base.encoder.in_channels),'icaf')
        affine=json.loads(Path(calibration).read_text(encoding='utf-8'))['affine']
        base.mote_fusion.geo=Sampler(True,affine)
        original=base.mote_fusion.fuse
        base.mote_fusion.fuse=lambda r,t,m:original(r,[base.mote_fusion.geo(a,b) for a,b in zip(r,t)],m)
        attach_hooks(base)
    else:
        base.rgbt_enabled=False;base.rgbt_freeze_thermal_stream=False;base.mote_enabled=False
        base.mote_fusion=None;base.thermal_backbone=None;base.thermal_encoder=None
    provenance={}
    if stage in ('RGB','IR'):
        if not official:raise ValueError('--official required for RGB/IR initialization')
        provenance['official']=transfer(base,state(official),('backbone.','encoder.','decoder.'))
    elif stage=='FUSION':
        if not rgb:raise ValueError('--rgb required (dataset-local RGB checkpoint)')
        provenance['rgb']=transfer(base,state(rgb),('backbone.','encoder.','decoder.'),strict=True)
        if fusion:
            if not ir:raise ValueError('--ir required (dataset-local IR checkpoint)')
            provenance['ir']=transfer(base,state(ir),('thermal_backbone.','thermal_encoder.'),strict=True)
    else:
        if not parent:raise ValueError('--parent required for REFINE/JOINT')
        saved=torch.load(parent,map_location='cpu')
        if saved.get('protocol',{}).get('arm',arm)!=arm:raise ValueError('Parent model configuration mismatch')
        base.load_state_dict(saved['base'],strict=True);refiner.load_state_dict(saved['refiner'],strict=True)
    if stage=='JOINT' and sbr:
        context=SAMSideSearchContext(refiner,base.encoder.in_channels[0])
        missing,unexpected=context.load_state_dict(refiner.state_dict(),strict=False)
        assert not unexpected and all(k.startswith(('context_projection.','context_score.')) for k in missing)
        refiner=context
    base.sgc_enabled=bool(sgc);base.sgc_aux_weight=10.
    base.sgc_decay_start,base.sgc_decay_end=(9,14) if stage=='FUSION' else (-1,-1)
    base.requires_grad_(False);refiner.requires_grad_(stage in ('REFINE','JOINT'))
    if stage!='REFINE':
        for module in (base.backbone,base.encoder,base.decoder):
            for n,p in module.named_parameters():
                if p.is_floating_point() and n not in ('up','reg_scale'):p.requires_grad_(True)
    if stage=='FUSION' and fusion:base.mote_fusion.requires_grad_(True)
    if stage=='REFINE' and not sbr:raise ValueError('Skip REFINE when SBR is disabled')
    s16={};base.encoder.register_forward_pre_hook(lambda m,a:s16.update(value=a[0][0].detach()))
    return base.to(device),refiner.to(device),cfg.criterion.to(device),s16,provenance

def modes(base,refiner,stage):
    base.train(stage!='REFINE');refiner.train(stage in ('REFINE','JOINT'))
    for name in ('thermal_backbone','thermal_encoder','mote_fusion'):
        m=getattr(base,name,None)
        if m is not None and (name!='mote_fusion' or stage!='FUSION'):m.eval()
    for m in base.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()

def optimizer(base,refiner,stage):
    if stage=='REFINE':groups=[dict(params=list(refiner.parameters()),lr=1e-4)]
    elif stage=='JOINT':groups=[dict(params=[p for p in base.parameters() if p.requires_grad],lr=1e-5),dict(params=list(refiner.parameters()),lr=5e-5)]
    else:groups=[dict(params=[p for n,p in base.named_parameters() if p.requires_grad and n.startswith('backbone.')],lr=1e-4),dict(params=[p for n,p in base.named_parameters() if p.requires_grad and not n.startswith('backbone.')],lr=2e-4)]
    groups=[g for g in groups if g['params']]
    for g in groups:g['initial_lr']=g['lr']
    return torch.optim.AdamW(groups,weight_decay=1e-4,foreach=False)

def move(samples,targets,device):
    return samples.to(device),[{k:v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]

def inputs(base,samples,stage):
    return samples if samples.shape[1]==3 else samples[:,3:] if stage=='IR' else samples if base.rgbt_enabled else samples[:,:3]

def readout(base,refiner,s16,out,hw):
    cache=base.sqer.cache;base.sqer.cache={}
    args=[cache['s4'].float(),cache['s8'].float()]
    if hasattr(refiner,'context_score'):args.append(s16['value'].float())
    args.extend([cache['base_logits'].float(),out['pred_boxes'].float(),hw])
    s16.clear();return refiner(*args)

def loss_step(base,refiner,criterion,s16,samples,targets,stage,epoch):
    if stage=='REFINE':
        with torch.no_grad():out=base(inputs(base,samples,stage))
        losses={};detection=out['pred_logits'].sum()*0.
    else:
        out=base(inputs(base,samples,stage),targets)
        losses=criterion(out,targets,epoch=max(epoch-2,0) if stage=='FUSION' else epoch)
        detection=sum(losses.values())
    localization=detection*0.;shape=0.
    if stage in ('REFINE','JOINT') and hasattr(refiner,'score'):
        result=readout(base,refiner,s16,out,samples.shape[-2:])
        matches=criterion.matcher(dict(pred_logits=out['base_pred_logits'],pred_boxes=out['pred_boxes']),targets)['indices']
        localization,stats=refiner.loss(result,out['pred_boxes'],targets,matches,shape_weight=1.)
        shape=stats['shape_loss']
    base.sqer.cache={};s16.clear()
    return detection+localization,dict(detection=float(detection.detach()),sgc=float(losses.get('loss_sgc_group',detection*0).detach()),sbr=float(localization.detach()),occupancy=float(shape))

@torch.no_grad()
def evaluate(cfg,base,refiner,s16,stage,device,confidence=.25,max_batches=None):
    base.eval();refiner.eval();post=cfg.postprocessor.to(device).eval()
    evaluator=cfg.evaluator;evaluator.cleanup();predictions={};tp=fp=fn=0;hook=None
    if hasattr(refiner,'context_score'):hook=refiner.context_score.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
    try:
        for i,(samples,targets) in enumerate(cfg.val_dataloader):
            if max_batches is not None and i>=max_batches:break
            samples,targets=move(samples,targets,device);out=base(inputs(base,samples,stage))
            boxes=out['pred_boxes']
            if stage in ('REFINE','JOINT') and hasattr(refiner,'score'):boxes=readout(base,refiner,s16,out,samples.shape[-2:])['boxes']
            results=post(dict(pred_boxes=boxes,pred_logits=out['pred_logits']),torch.stack([t['orig_size'] for t in targets]))
            batch={int(t['image_id']):{k:v.cpu() for k,v in r.items()} for t,r in zip(targets,results)}
            evaluator.update(batch);predictions.update(batch)
            for image_id,p in batch.items():
                truth=torch.tensor([a['bbox'] for a in cfg.val_dataloader.dataset.coco.imgToAnns.get(image_id,[])],dtype=torch.float32).reshape(-1,4);truth[:,2:]+=truth[:,:2]
                keep=p['scores']>=confidence;b=p['boxes'][keep][p['scores'][keep].argsort(descending=True)];used=set()
                for row in box_iou(b,truth):
                    options=[j for j in range(len(truth)) if j not in used and float(row[j])>=.5]
                    if options:used.add(max(options,key=lambda j:float(row[j])))
                tp+=len(used);fp+=len(b)-len(used);fn+=len(truth)-len(used)
            base.sqer.cache={};s16.clear()
        evaluator.synchronize_between_processes();evaluator.accumulate();evaluator.summarize()
        p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
        return dict(coco_eval_bbox=evaluator.coco_eval['bbox'].stats.tolist(),precision=p,recall=r,f1=2*p*r/max(p+r,1e-12),confidence_threshold=confidence,pr_iou_threshold=.5,images=len(predictions)),predictions
    finally:
        if hook is not None:hook.remove()

def factor(stage,progress,epochs):
    if stage in ('RGB','IR','FUSION'):
        return max(.01,progress/2) if progress<2 else .1+.9*.5*(1+math.cos(math.pi*min(1.,(progress-2)/max(epochs-2,1))))
    if stage=='REFINE':return max(progress,.01) if progress<1 else .01+.99*.5*(1+math.cos(math.pi*min(1.,(progress-1)/max(epochs-1,1))))
    return .1+.9*.5*(1+math.cos(math.pi*min(1.,progress/max(epochs,1))))
