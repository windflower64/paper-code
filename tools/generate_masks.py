"""Generate train-only SAM3 teachers using the accepted box/jitter protocol."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/max(union,1e-12)

def bounds(mask):
    y,x=np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else [0,0,0,0]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--limit',type=int,default=0,help='Smoke limit; does not produce complete records.json')
    a=p.parse_args();root=Path(a.data_root);out=root/'sam_masks/train'
    data=json.loads((root/'annotations/train.json').read_text());annotations=defaultdict(list)
    for ann in data['annotations']:annotations[ann['image_id']].append(ann)
    if any(len(v)>1 for v in annotations.values()):raise ValueError('This release uses the original single-target SBR/mask protocol')
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(4);torch.manual_seed(20260928)
    model=build_sam3_image_model(checkpoint_path=a.checkpoint,load_from_HF=False,compile=False)
    processor=Sam3Processor(model,confidence_threshold=.2)
    (out/'masks').mkdir(parents=True,exist_ok=True);(out/'records_by_id').mkdir(exist_ok=True)
    for index,im in enumerate(sorted(data['images'],key=lambda r:r['id'])):
        if a.limit and index>=a.limit:break
        rowfile=out/'records_by_id'/f"{im['id']:06d}.json"
        if rowfile.exists():continue
        record=dict(image_id=im['id'],file_name=im['file_name'],accepted=False,supervision_weight=0.)
        if annotations[im['id']]:
            x,y,w,h=annotations[im['id']][0]['bbox'];gt=[x,y,x+w,y+h]
            image=Image.open(root/'train/visible/images'/im['file_name']).convert('RGB');width,height=image.size
            if (width,height)!=(im['width'],im['height']):raise ValueError('Image annotation size mismatch')
            masks=[];available=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=processor.set_image(image)
                for sign in (0,1,-1):
                    dx=sign*max(1.,.05*w);dy=sign*max(1.,.05*h)
                    x0,x1=np.clip([x+dx,x+w+dx],0,width);y0,y1=np.clip([y+dy,y+h+dy],0,height)
                    box=[float((x0+x1)/(2*width)),float((y0+y1)/(2*height)),float((x1-x0)/width),float((y1-y0)/height)]
                    processor.reset_all_prompts(state);processor.set_text_prompt(prompt='drone',state=state)
                    pred=processor.add_geometric_prompt(box=box,label=True,state=state)
                    boxes=pred['boxes'].float().cpu().tolist();available.append(bool(boxes))
                    selected=max(range(len(boxes)),key=lambda k:iou(boxes[k],gt)) if boxes else None
                    masks.append(pred['masks'].cpu().numpy()[selected,0].astype(bool) if boxes else np.zeros((height,width),dtype=bool))
            overlap=iou(bounds(masks[0]),gt)
            stability=min(float((masks[0]&m).sum()/max(1,(masks[0]|m).sum())) for m in masks[1:])
            accepted=bool(all(available) and overlap>=.5 and stability>=.7)
            record.update(accepted=accepted,supervision_weight=float(accepted),mask_bbox_iou=overlap,
                prompt_mask_stability=stability,bbox_xyxy_annotation=gt)
            if accepted:Image.fromarray(masks[0].astype(np.uint8)*255).save(out/'masks'/f"{im['id']:06d}.png")
        rowfile.write_text(json.dumps(record),encoding='utf-8');print(index+1,record,flush=True)
    if not a.limit:
        records=[json.loads((out/'records_by_id'/f"{im['id']:06d}.json").read_text()) for im in sorted(data['images'],key=lambda r:r['id'])]
        (out/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':main()
