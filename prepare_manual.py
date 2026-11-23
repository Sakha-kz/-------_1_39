"""Prepare manually acquired datasets without deleting or moving originals."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image

def imagefolder(source,destination,seed):
    source=Path(source).resolve(); destination=Path(destination).resolve()
    if source==destination or source in destination.parents: raise ValueError('Destination must be outside source')
    rng=np.random.default_rng(seed); manifest=[]
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        files=[p for p in sorted(folder.iterdir()) if p.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff','.bmp']]
        rng.shuffle(files); boundary=int(.8*len(files))
        for i,p in enumerate(files):
            split='train' if i<boundary else 'test'; out=destination/split/folder.name/(p.stem+'.png'); out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
            with Image.open(p) as im: im.convert('RGB').save(out)
            manifest.append(dict(source=str(p),destination=str(out),split=split))
    (destination/'import_manifest.json').write_text(json.dumps(dict(seed=seed,files=manifest),indent=2),encoding='utf-8')

def quickdraw(source,destination,seed,per_class=10000):
    source=Path(source); files=sorted(source.glob('*.npy')); rng=np.random.default_rng(seed)
    if len(files)<10: raise ValueError('Supply at least 10 official QuickDraw bitmap .npy files')
    chosen=rng.choice(len(files),10,replace=False); dest=Path(destination)
    for j in chosen:
        p=files[int(j)]; data=np.load(p,mmap_mode='r'); ids=rng.choice(len(data),min(per_class,len(data)),replace=False)
        for i,idx in enumerate(ids):
            split='train' if i<int(.8*len(ids)) else 'test'; out=dest/split/p.stem/f'{idx}.png'; out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists(): raise FileExistsError(str(out))
            Image.fromarray(np.asarray(data[idx]).reshape(28,28).astype('uint8')).save(out)
    (dest/'selection.json').write_text(json.dumps(dict(seed=seed,files=[files[int(j)].name for j in chosen],maximum_per_class=per_class),indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['imagefolder','quickdraw']); p.add_argument('source'); p.add_argument('destination'); p.add_argument('--seed',type=int,default=1017); p.add_argument('--per-class',type=int,default=10000); a=p.parse_args()
    if a.mode=='imagefolder': imagefolder(a.source,a.destination,a.seed)
    else: quickdraw(a.source,a.destination,a.seed,a.per_class)
