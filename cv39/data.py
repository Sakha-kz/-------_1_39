"""Raw datasets, persisted splits, train-only normalisation and transforms."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets as D,transforms as T
from torchvision.transforms import functional as TF
from PIL import Image,ImageDraw
from sklearn.model_selection import train_test_split
from .util import dump

class Shapes(Dataset):
    classes=['circle','square','triangle']
    def __init__(self,n=6000,seed=42,domain='normal'):
        self.n=n; self.seed=seed; self.domain=domain; self.targets=[i%3 for i in range(n)]
    def __len__(self): return self.n
    def __getitem__(self,i):
        rng=np.random.default_rng(self.seed+i); c=self.targets[i]; im=Image.new('RGB',(32,32),(15,15,15)); draw=ImageDraw.Draw(im)
        easy=self.domain=='easy'; r=10 if easy else int(rng.integers(5,11)); cx=16 if easy else int(rng.integers(r,33-r)); cy=16 if easy else int(rng.integers(r,33-r))
        colors=[(220,60,60),(60,220,60),(60,60,220)]
        color=(220,190,50) if self.domain=='newcolor' else colors[int(rng.integers(3))]
        if c==0: draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=color)
        elif c==1: draw.rectangle((cx-r,cy-r,cx+r,cy+r),fill=color)
        else: draw.polygon([(cx,cy-r),(cx-r,cy+r),(cx+r,cy+r)],fill=color)
        return im,c

class UTK(Dataset):
    classes=['0-12','13-19','20-35','36-55','56+']
    def __init__(self,root):
        self.samples=[]; self.groups=[]
        for p in sorted(Path(root).rglob('*.jpg')):
            parts=p.name.split('_')
            try: age,gender,race=map(int,parts[:3])
            except (ValueError,TypeError): continue
            if not (0<=age<=116 and gender in [0,1] and 0<=race<=4): continue
            self.samples.append((p,int(np.digitize(age,[13,20,36,56])))); self.groups.append(f'{gender}_{race}')
        if not self.samples: raise FileNotFoundError('Place original UTKFace files in data/utkface')
        self.targets=[s[1] for s in self.samples]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,y=self.samples[i]; return Image.open(p).convert('RGB'),y

class Patterns(Dataset):
    classes=['stripes','checkerboard','dots']
    def __init__(self,n=1800,seed=42): self.n=n; self.seed=seed; self.targets=[i%3 for i in range(n)]
    def __len__(self): return self.n
    def __getitem__(self,i):
        rng=np.random.default_rng(self.seed+i); y,x=np.mgrid[:128,:128]; k=int(rng.integers(4,14)); c=self.targets[i]
        mask=((x//k)%2==0) if c==0 else (((x//k+y//k)%2==0) if c==1 else ((x%k-k//2)**2+(y%k-k//2)**2<(k/3)**2))
        image=np.zeros((128,128,3),dtype=np.uint8); image[:]=rng.integers(0,60,(3,)); image[mask]=rng.integers(150,256,(3,))
        return Image.fromarray(image),c

class LabelView(Dataset):
    def __init__(self,base,labels,classes=None):
        self.base=base; self.targets=list(map(int,labels)); self.classes=classes or [str(i) for i in sorted(set(self.targets))]
    def __len__(self): return len(self.base)
    def __getitem__(self,i): return self.base[i][0],self.targets[i]

def labels(ds):
    if hasattr(ds,'targets'): return np.asarray(ds.targets,dtype=int)
    if hasattr(ds,'labels'): return np.asarray(ds.labels,dtype=int)
    if hasattr(ds,'_labels'): return np.asarray(ds._labels,dtype=int)
    if hasattr(ds,'_samples'): return np.asarray([s[1] for s in ds._samples])
    return np.asarray([ds[i][1] for i in range(len(ds))])

def raw_datasets(cfg):
    root=Path(cfg['root']); name=cfg['dataset']; kw=dict(root=str(root),download=True)
    if name=='synthetic': return Shapes(seed=cfg['data_seed']),None,Shapes(1200,cfg['data_seed']+100000)
    if name=='patterns': return Patterns(seed=cfg['data_seed']),None,Patterns(600,cfg['data_seed']+100000)
    if name in ['mnist','fashion_mnist','kmnist','emnist','cifar10','cifar100']:
        factory={'mnist':D.MNIST,'fashion_mnist':D.FashionMNIST,'kmnist':D.KMNIST,'emnist':D.EMNIST,'cifar10':D.CIFAR10,'cifar100':D.CIFAR100}[name]
        if name=='emnist': kw['split']='letters'
        tr,te=factory(train=True,**kw),factory(train=False,**kw)
        if name=='emnist':
            tr=LabelView(tr,labels(tr)-1,[chr(65+i) for i in range(26)]); te=LabelView(te,labels(te)-1,tr.classes)
        if name=='cifar100':
            # Trusted torchvision-downloaded CIFAR files, archive integrity checked upstream.
            import pickle
            folder=root/'cifar-100-python'; coarse=[]
            for split in ['train','test']:
                with (folder/split).open('rb') as f: coarse.append(pickle.load(f,encoding='bytes')[b'coarse_labels'])
            tr=LabelView(tr,coarse[0],[f'coarse_{i}' for i in range(20)]); te=LabelView(te,coarse[1],tr.classes)
        return tr,None,te
    if name=='svhn': return D.SVHN(split='train',**kw),None,D.SVHN(split='test',**kw)
    if name=='stl10': return D.STL10(split='train',**kw),None,D.STL10(split='test',**kw)
    if name=='flowers102': return tuple(D.Flowers102(split=s,**kw) for s in ['train','val','test'])
    if name=='dtd': return tuple(D.DTD(split=s,**kw) for s in ['train','val','test'])
    if name=='oxford_pets': return D.OxfordIIITPet(split='trainval',**kw),None,D.OxfordIIITPet(split='test',**kw)
    if name=='food101': return D.Food101(split='train',**kw),None,D.Food101(split='test',**kw)
    if name=='gtsrb': return D.GTSRB(split='train',**kw),None,D.GTSRB(split='test',**kw)
    if name=='eurosat': return D.EuroSAT(**kw),None,None
    if name=='caltech101': return D.Caltech101(**kw),None,None
    if name=='utkface': return UTK(root/'utkface'),None,None
    if name in ['ucmerced','quickdraw']:
        p=root/name
        if not (p/'train').exists(): raise FileNotFoundError(f'Place {name}/train/<class> and test/<class> in {root}')
        tr=D.ImageFolder(p/'train'); te=D.ImageFolder(p/'test')
        if tr.class_to_idx!=te.class_to_idx: raise ValueError('Train/test class folders differ')
        return tr,None,te
    raise ValueError(name)

class Noise:
    def __call__(self,x): return (x+.05*torch.randn_like(x)).clamp(0,1)

def transform(cfg,train=False,stats=None):
    size=cfg['size']; aug=cfg['aug'] if train else 'none'; ops=[T.Resize((size,size)),T.Grayscale(3) if cfg.get('grayscale') else T.Lambda(lambda im:im.convert('RGB'))]
    if aug in ['basic','digit_crop','sign_crop']: ops += [T.RandomCrop(size,padding=max(2,size//8))]
    if aug=='basic': ops += [T.RandomHorizontalFlip()]
    if aug in ['digit_crop','sign_crop']: ops += [T.ColorJitter(brightness=.2,contrast=.2,saturation=.2)]
    if aug.startswith('rotate'): ops += [T.RandomRotation(int(aug[6:]))]
    if aug=='strong': ops += [T.RandomResizedCrop(size,scale=(.7,1)),T.RandomHorizontalFlip(),T.RandAugment()]
    if aug=='texture': ops += [T.RandomResizedCrop(size,scale=(.3,1))]
    if aug=='corruption': ops += [T.RandomApply([T.GaussianBlur(5)],p=.5),T.ColorJitter(brightness=.3)]
    ops += [T.ToTensor()]
    if aug in ['noise','corruption']: ops += [Noise()]
    mean,std=([.485,.456,.406],[.229,.224,.225]) if cfg['norm']=='imagenet' else stats
    ops += [T.Normalize(mean,std)]
    return T.Compose(ops)

class Wrapped(Dataset):
    def __init__(self,base,indices,mapping,tf):
        self.base=base; self.indices=list(map(int,indices)); self.mapping=mapping; self.transform=tf
        all_y=labels(base); self.targets=[mapping[int(all_y[i])] for i in self.indices]
        names=getattr(base,'classes',None)
        self.classes=[str(names[c]) if names and c<len(names) else str(c) for c in sorted(mapping,key=mapping.get)]
    def __len__(self): return len(self.indices)
    def raw(self,i):
        im,_=self.base[self.indices[i]]; return im.convert('RGB'),self.targets[i]
    def __getitem__(self,i):
        im,y=self.raw(i); return self.transform(im),y

def split_indices(y,seed,fraction):
    indices=np.arange(len(y)); a,b=train_test_split(indices,test_size=fraction,random_state=seed,stratify=y)
    return a,b

def prepare(cfg,folder):
    tr,va,te=raw_datasets(cfg); seed=cfg['data_seed']; y=labels(tr)
    cs=sorted(map(int,set(y))); count=10 if cfg['dataset'] in ['cifar100','quickdraw'] else cfg.get('subset')
    if count and len(cs)>count: cs=sorted(np.random.default_rng(seed).choice(cs,count,replace=False).tolist())
    mapping={c:i for i,c in enumerate(cs)}; selected=np.flatnonzero(np.isin(y,cs))
    if te is None:
        a,b=split_indices(y[selected],seed,.2); train_ids=selected[a]; rest=selected[b]
        a,b=split_indices(y[rest],seed,.5); val_ids,test_ids=rest[a],rest[b]; va=te=tr
    elif va is None:
        a,b=split_indices(y[selected],seed,.1); train_ids,val_ids=selected[a],selected[b]; va=tr
        test_ids=np.flatnonzero(np.isin(labels(te),cs))
    else:
        train_ids=selected; val_ids=np.flatnonzero(np.isin(labels(va),cs)); test_ids=np.flatnonzero(np.isin(labels(te),cs))
    if cfg.get('train_limit'): train_ids=train_ids[:cfg['train_limit']]
    if cfg.get('val_limit'): val_ids=val_ids[:cfg['val_limit']]
    # Streaming moments over TRAIN only, evaluated at the chosen input size.
    s=np.zeros(3); ss=np.zeros(3); pixels=0
    for idx in train_ids:
        im=np.asarray(tr[int(idx)][0].convert('RGB').resize((cfg['size'],cfg['size'])),dtype=np.float64)/255
        s+=im.sum((0,1)); ss+=(im*im).sum((0,1)); pixels+=im.shape[0]*im.shape[1]
    mean=s/pixels; std=np.maximum(np.sqrt(np.maximum(ss/pixels-mean*mean,0)),1e-6)
    stats=(mean.tolist(),std.tolist())
    data={'train':Wrapped(tr,train_ids,mapping,transform(cfg,True,stats)),
          'val':Wrapped(va,val_ids,mapping,transform(cfg,False,stats)),
          'test':Wrapped(te,test_ids,mapping,transform(cfg,False,stats)),'stats':stats,'mapping':mapping}
    manifest=dict(dataset=cfg['dataset'],seed=seed,selected_classes=cs,classes=data['train'].classes,
                  train=list(map(int,train_ids)),val=list(map(int,val_ids)),test=list(map(int,test_ids)),stats=stats,
                  official_test=te is not tr,smoke=cfg.get('smoke',False))
    manifest['sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    dump(Path(folder)/'split.json',manifest)
    return data
