"""Download the D-FINE-N COCO checkpoint with an expected SHA-256 guard."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL='https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth'
EXPECTED='41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='weights/dfine_n_coco.pth');a=p.parse_args()
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=EXPECTED:raise ValueError('Existing file has a different SHA; choose a new output path')
    else:
        temp=target.with_suffix('.download')
        with urlopen(URL,timeout=60) as response,temp.open('wb') as f:
            for chunk in iter(lambda:response.read(1024**2),b''):f.write(chunk)
        if digest(temp)!=EXPECTED:raise ValueError('Downloaded checkpoint SHA differs; original release may have changed')
        temp.replace(target)
    print('PRETRAINED_CHECKPOINT_OK',target)
