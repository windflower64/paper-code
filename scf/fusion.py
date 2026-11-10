"""Current fusion runtime extracted from the accepted research implementation.
ICAFusion interaction is credited and distributed under AGPL-3.0.
"""
import torch
from torch import nn
from torch.nn import functional as F
from third_party.icafusion.blocks import TransformerFusionBlock

def author_block():
    return TransformerFusionBlock

class PreEncoderFusion(nn.Module):
    def __init__(self, channels, kind):
        super().__init__()
        self.kind = kind
        self.blocks = nn.ModuleList()
        for c in channels:
            if kind == 'concat':
                layer = nn.Conv2d(c * 2, c, 1, bias=False)
                with torch.no_grad():
                    layer.weight.zero_()
                    layer.weight[:, :c, 0, 0].copy_(torch.eye(c))
                self.blocks.append(layer)
            else:
                self.blocks.append(ICAFAdapter(c))
        self.bypass = False
        self.amp_safe = False
        self.last_gate_mean = torch.tensor(1.)
        self.last_candidate_scores = torch.ones(1, 1)
        self.last_object_attention_mean = torch.tensor(1.)
        self.last_update_ratio = torch.tensor(0.)

    def fuse(self, rgb, ir, mask):
        assert len(rgb) == len(ir) == len(self.blocks) == 2
        outputs, ratios = [], []
        for r, t, block in zip(rgb, ir, self.blocks):
            assert r.shape == t.shape
            if self.bypass:
                y = r
            else:
                raw = block(torch.cat((r, t), 1)) if self.kind == 'concat' else block(r, t)
                y = torch.where(mask[:, None, None, None], raw, r)
            outputs.append(y)
            ratios.append(((y-r).detach().square().mean((1,2,3)).sqrt()
                / r.detach().square().mean((1,2,3)).sqrt().clamp_min(1e-6)).mean())
        self.last_update_ratio = torch.stack(ratios).mean()
        self.last_gate_mean = mask.float().mean()
        self.last_candidate_scores = mask[:, None].float()
        return outputs

    def forward(self, visible, *args, **kwargs):
        # Historical post-encoder slot is intentionally a no-op.
        return visible, None

class ICAFAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rgb_projection = nn.Conv2d(channels, 128, 1)
        self.ir_projection = nn.Conv2d(channels, 128, 1)
        self.interaction = author_block()(128, vert_anchors=16, horz_anchors=16,
            h=8, n_layer=1)
        self.output = nn.Conv2d(128, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, rgb, ir):
        features = self.interaction((self.rgb_projection(rgb), self.ir_projection(ir)))
        return rgb + self.output(features)

def attach_hooks(model):
    cache = {}
    def capture_input(module, args):
        cache.clear()
        cache['mask'] = args[0][:, 3:].detach().abs().amax((1,2,3)) > 1e-8
    def capture_ir(module, args):
        cache['ir'] = args[0]
    def fuse_input(module, args):
        ir = cache.pop('ir')
        return (model.mote_fusion.fuse(args[0], ir, cache['mask']),)
    model.register_forward_pre_hook(capture_input)
    model.thermal_encoder.register_forward_pre_hook(capture_ir)
    model.encoder.register_forward_pre_hook(fuse_input)

class Sampler(nn.Module):
    def __init__(self,adaptive,affine):
        super().__init__();w=torch.tensor(affine,dtype=torch.float32)
        self.register_buffer('matrix',w[:2]);self.register_buffer('translation',w[2]);self.adaptive=adaptive
        if adaptive:
            self.offset=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,2,3,padding=1))
            nn.init.zeros_(self.offset[-1].weight);nn.init.zeros_(self.offset[-1].bias)
    def forward(self,r,t):
        b,_,h,w=r.shape
        yy,xx=torch.meshgrid((torch.arange(h,device=r.device)+.5)/h,(torch.arange(w,device=r.device)+.5)/w,indexing='ij')
        uv=torch.stack((xx,yy),-1);ir=(uv-self.translation)@torch.linalg.inv(self.matrix)
        grid=(ir*2-1).unsqueeze(0).expand(b,-1,-1,-1)
        base=F.grid_sample(t,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
        if not self.adaptive:return base
        def response(x):
            z=x.square().mean(1,keepdim=True).sqrt();return z/z.mean((2,3),keepdim=True).clamp_min(1e-6)
        delta=self.offset(torch.cat((response(r),response(base)),1)).tanh().permute(0,2,3,1)
        # Maximum residual: two IR feature cells in either axis.
        delta=delta*delta.new_tensor([4/t.shape[-1],4/t.shape[-2]])
        return F.grid_sample(t,grid+delta,mode='bilinear',padding_mode='zeros',align_corners=False)
