import json,random,os,time,platform
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import numpy as np
import torch

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8'); os.replace(temp,path)

def atomic_save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp'); torch.save(obj,temp); os.replace(temp,path)

def environment():
    packages={}
    for p in ['torch','torchvision','numpy','scikit-learn','timm','transformers','scikit-image']:
        try: packages[p]=version(p)
        except PackageNotFoundError: packages[p]=None
    return dict(python=platform.python_version(),packages=packages,cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def fewshot(targets,k,seed):
    rng=np.random.default_rng(seed); labels=np.asarray(targets); ids=[]
    for c in np.unique(labels):
        group=np.flatnonzero(labels==c); ids.extend(rng.choice(group,min(k,len(group)),replace=False).tolist())
    return ids

def longtail(targets,ratio,seed):
    rng=np.random.default_rng(seed); y=np.asarray(targets); cs=np.unique(y); max_n=min(np.sum(y==c) for c in cs); ids=[]
    for i,c in enumerate(cs):
        n=max(1,int(max_n*ratio**(-i/max(len(cs)-1,1))))
        ids.extend(rng.choice(np.flatnonzero(y==c),n,replace=False).tolist())
    return ids
