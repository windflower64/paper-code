"""Inference assembly matching the exported, accepted checkpoints."""
import json
from pathlib import Path
import torch
from torch import nn
from src.core import YAMLConfig
from src.zoo.dfine.sam_side_search import SAMSideSearch
from src.zoo.dfine.sam_side_search_multiscale import SAMSideSearchContext
from .fusion import PreEncoderFusion, Sampler, attach_hooks


class FeatureTap(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache={}

    def forward(self,s4,s8,queries,logits,boxes,**kwargs):
        self.cache=dict(s4=s4.detach(),s8=s8.detach(),base_logits=logits.detach())
        return dict(delta_logits=torch.zeros_like(logits),
            query_indices=logits.sigmoid().amax(-1).topk(min(64,logits.shape[1]),dim=1).indices,
            roi_grid=None,attention=None)


class Detector(nn.Module):
    def __init__(self,config,checkpoint,calibration):
        super().__init__()
        cfg=YAMLConfig(str(config))
        self.base=cfg.model
        saved=torch.load(checkpoint,map_location='cpu')
        self.metadata=saved.get('metadata',{})
        self.stage=self.metadata.get('stage',saved.get('protocol',{}).get('stage','JOINT'))
        from .training import ARMS
        self.arm=saved.get('protocol',{}).get('arm','FULL')
        sgc,sbr,fusion=ARMS[self.arm]
        if self.stage in ('RGB','IR'):sgc=sbr=fusion=0
        reader=self.base.sqer
        if not sbr:
            self.refiner=nn.Module()
        elif any(k.startswith('context_projection.') for k in saved['refiner']):
            self.refiner=SAMSideSearchContext(reader,self.base.encoder.in_channels[0])
            # Accepted JOINT metrics explicitly disable S16 context scoring.
            self.refiner.context_score.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
        else:
            self.refiner=SAMSideSearch(reader)
        self.base.sqer=FeatureTap()
        if fusion:
            self.base.mote_fusion=PreEncoderFusion(list(self.base.encoder.in_channels),'icaf')
            affine=json.loads(Path(calibration).read_text())['affine']
            self.base.mote_fusion.geo=Sampler(True,affine)
            original=self.base.mote_fusion.fuse
            self.base.mote_fusion.fuse=lambda rgb,ir,mask:original(rgb,[self.base.mote_fusion.geo(r,t) for r,t in zip(rgb,ir)],mask)
            attach_hooks(self.base)
        else:
            self.base.rgbt_enabled=False;self.base.rgbt_freeze_thermal_stream=False;self.base.mote_enabled=False
            self.base.mote_fusion=None;self.base.thermal_backbone=None;self.base.thermal_encoder=None
        self.base.sgc_enabled=bool(sgc)
        self.base.load_state_dict(saved['base'],strict=True)
        self.refiner.load_state_dict(saved['refiner'],strict=True)
        self.postprocessor=cfg.postprocessor
        self.s16={}
        self.base.encoder.register_forward_pre_hook(lambda m,a:self.s16.update(value=a[0][0].detach()))
        self.eval()

    @torch.no_grad()
    def forward(self,images,original_sizes):
        """images: Bx6x512x640, RGB then 3-channel IR, float [0,1].

        original_sizes: Bx2 tensor [original RGB width, original RGB height].
        Preprocess native RGB and IR independently to the common canvas.
        """
        assert images.ndim==4 and images.shape[1] in (3,6) and images.shape[-2:]==(512,640)
        x=images[:,3:] if self.stage=='IR' and images.shape[1]==6 else images if self.base.rgbt_enabled else images[:,:3]
        out=self.base(x)
        cache=self.base.sqer.cache
        if self.stage not in ('REFINE','JOINT') or not hasattr(self.refiner,'score'):
            result=dict(boxes=out['pred_boxes'])
        elif hasattr(self.refiner,'context_score'):
            result=self.refiner(cache['s4'].float(),cache['s8'].float(),self.s16['value'].float(),cache['base_logits'].float(),out['pred_boxes'].float(),images.shape[-2:])
        else:
            result=self.refiner(cache['s4'].float(),cache['s8'].float(),cache['base_logits'].float(),out['pred_boxes'].float(),images.shape[-2:])
        self.base.sqer.cache={};self.s16.clear()
        return self.postprocessor(dict(pred_boxes=result['boxes'],pred_logits=out['pred_logits']),original_sizes)
