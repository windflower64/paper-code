"""Validate public split layout; derive IR COCO labels and train-only calibration."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

def prepare(root,output,require_masks=True):
    root=Path(root);all_names={};thermal_centers=[];visible_centers=[]
    for split in ('train','val','test'):
        data=json.loads((root/'annotations'/f'{split}.json').read_text(encoding='utf-8'))
        if {int(c['id']) for c in data['categories']}!={0}:raise ValueError('Single UAV class must use category_id=0')
        groups=defaultdict(list)
        for a in data['annotations']:
            if a['category_id']!=0:raise ValueError('Unexpected category')
            groups[a['image_id']].append(a)
        ir_images=[];annotations=[]
        for im in data['images']:
            name=im['file_name']
            if name in all_names:raise ValueError(f'{name} appears in both {split} and {all_names[name]}')
            all_names[name]=split
            for modality in ('visible','thermal'):
                path=root/split/modality/'images'/name
                if not path.is_file():raise FileNotFoundError(path)
            label=root/split/'thermal/labels'/f'{Path(name).stem}.txt'
            if not label.is_file():raise FileNotFoundError(label)
            with Image.open(root/split/'thermal/images'/name) as image:w,h=image.size
            ir_images.append(dict(id=im['id'],file_name=name,width=w,height=h))
            rows=[line.split() for line in label.read_text().splitlines() if line.strip()]
            for row in rows:
                category=int(row[0]);x,y,bw,bh=map(float,row[1:5])
                if category!=0 or not all(np.isfinite([x,y,bw,bh])) or not (0<=x<=1 and 0<=y<=1 and 0<bw<=1 and 0<bh<=1):raise ValueError(label)
                annotations.append(dict(id=len(annotations)+1,image_id=im['id'],category_id=0,bbox=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h],area=bw*w*bh*h,iscrowd=0))
            if split=='train' and len(groups[im['id']])==len(rows)==1:
                x,y,w0,h0=groups[im['id']][0]['bbox']
                thermal_centers.append([float(rows[0][1]),float(rows[0][2]),1.])
                visible_centers.append([(x+w0/2)/im['width'],(y+h0/2)/im['height']])
        target=root/'annotations'/f'ir_{split}.json'
        payload=dict(images=ir_images,annotations=annotations,categories=data['categories'])
        if target.exists() and json.loads(target.read_text())!=payload:raise FileExistsError(f'Refuse replacing different IR annotations: {target}')
        target.write_text(json.dumps(payload),encoding='utf-8')
        print(split,len(ir_images),'pairs',len(annotations),'IR instances',flush=True)
    if len(thermal_centers)<3:raise ValueError('At least three unambiguous TRAIN pairs are needed for affine calibration')
    A=np.asarray(thermal_centers);B=np.asarray(visible_centers)
    if np.linalg.matrix_rank(A)<3:raise ValueError('Degenerate train calibration centers')
    matrix=np.linalg.lstsq(A,B,rcond=None)[0]
    if abs(np.linalg.det(matrix[:2]))<.01:raise ValueError('Singular calibration')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(dict(affine=matrix.tolist(),source='unambiguous train annotation centers',pairs=len(A)),indent=2),encoding='utf-8')
    if require_masks:
        records=json.loads((root/'sam_masks/train/records.json').read_text())
        train=json.loads((root/'annotations/train.json').read_text())
        mapping={im['id']:im['file_name'] for im in train['images']}
        if {r['image_id'] for r in records}!=set(mapping):raise ValueError('Mask records must cover exactly training IDs')
        for r in records:
            if r.get('file_name',mapping[r['image_id']])!=mapping[r['image_id']]:raise ValueError('Mask filename mismatch')
            if r.get('accepted'):
                for name in r.get('instance_mask_files',[f"{r['image_id']:06d}.png"]):
                    if Path(name).name!=name or not (root/'sam_masks/train/masks'/name).is_file():raise FileNotFoundError(name)
    print('DATA_CHECK_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--calibration-output',required=True)
    p.add_argument('--without-masks',action='store_true');a=p.parse_args()
    prepare(a.data_root,a.calibration_output,not a.without_masks)
