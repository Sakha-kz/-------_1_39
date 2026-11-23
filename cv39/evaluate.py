import io,time,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image,ImageEnhance,ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from .train import predict
from .util import dump,sync

def metrics(logits,y,classes):
    probs=logits.softmax(-1); pred=probs.argmax(-1); n=len(y); confidence=probs.max(-1).values; correct=pred==y
    ece=0.
    for i in range(15):
        mask=(confidence>=i/15)&(confidence<((i+1)/15) if i<14 else confidence<=1)
        if mask.any(): ece+=mask.float().mean().item()*abs(confidence[mask].mean().item()-correct[mask].float().mean().item())
    cm=confusion_matrix(y,pred,labels=list(range(len(classes)))); normalized=cm/np.maximum(cm.sum(1,keepdims=True),1)
    pairs=[dict(true=classes[i],predicted=classes[j],count=int(cm[i,j])) for i,j in np.ndindex(cm.shape) if i!=j and cm[i,j]>0]
    pairs=sorted(pairs,key=lambda r:-r['count'])[:5]
    return dict(n=n,accuracy=correct.float().mean().item(),top3=(probs.topk(min(3,len(classes)),1).indices==y[:,None]).any(1).float().mean().item(),
        macro_F1=f1_score(y,pred,average='macro',zero_division=0),ECE=ece,
        per_class=classification_report(y,pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0),
        confusion=cm.tolist(),normalized_confusion=normalized.tolist(),top5_confusions=pairs)

class Corrupt(Dataset):
    def __init__(self,base,kind,seed=42): self.base=base; self.kind=kind; self.seed=seed
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        im,y=self.base.raw(i); kind=self.kind
        if kind.startswith('noise'):
            sigma=float(kind.split(':')[1]); z=np.array(im).astype(float)/255
            z+=np.random.default_rng(self.seed+i).normal(0,sigma,z.shape); im=Image.fromarray((z.clip(0,1)*255).astype('uint8'))
        elif kind=='blur': pass
        elif kind=='bright+': im=ImageEnhance.Brightness(im).enhance(1.3)
        elif kind=='bright-': im=ImageEnhance.Brightness(im).enhance(.7)
        elif kind=='jpeg':
            buf=io.BytesIO(); im.save(buf,format='JPEG',quality=20); buf.seek(0); im=Image.open(buf).convert('RGB')
        elif kind.startswith('shift:'):
            from torchvision.transforms import functional as TF
            delta=int(kind.split(':')[1]); im=TF.affine(im,angle=0,translate=[delta,delta],scale=1,shear=0)
        elif kind.startswith('scale:'):
            from torchvision.transforms import functional as TF
            im=TF.affine(im,angle=0,translate=[0,0],scale=float(kind.split(':')[1]),shear=0)
        if kind=='blur':
            from torchvision.transforms import functional as TF
            im=TF.gaussian_blur(im,[5,5],sigma=[1.,1.])
        return self.base.transform(im),y

def latency(model,ds,device):
    model=model.to(device).eval(); samples=[ds[i][0].unsqueeze(0).to(device) for i in range(min(100,len(ds)))]
    with torch.inference_mode():
        for i in range(20): model(samples[i%len(samples)])
        sync(device); start=time.perf_counter()
        for x in samples: model(x)
        sync(device); seconds=(time.perf_counter()-start)/len(samples)
    return dict(device=device,precision=str(next(model.parameters()).dtype),warmup=20,n=len(samples),milliseconds=seconds*1000,FPS=1/seconds)

def figures(data,folder):
    folder=Path(folder); classes=data['train'].classes; counts=np.bincount(data['train'].targets,minlength=len(classes))
    fig,ax=plt.subplots(figsize=(max(8,len(classes)*.25),4)); ax.bar(range(len(classes)),counts); ax.set(xlabel='Class index',ylabel='Train images'); fig.tight_layout(); fig.savefig(folder/'class_histogram.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(len(classes),3,figsize=(7,max(3,len(classes)*1.5)),squeeze=False)
    for c,name in enumerate(classes):
        ids=np.flatnonzero(np.asarray(data['train'].targets)==c)[:3]
        for ax,i in zip(axes[c],ids): ax.imshow(data['train'].raw(i)[0]); ax.set_title(name,fontsize=8); ax.axis('off')
        for ax in axes[c]: ax.axis('off')
    fig.tight_layout(); fig.savefig(folder/'contact_sheet.png',dpi=100); plt.close(fig)
    from skimage.filters import sobel,threshold_otsu
    from skimage.feature import canny
    from skimage.morphology import binary_opening,binary_closing,disk,skeletonize
    fig,axes=plt.subplots(3,7,figsize=(16,7))
    for row in range(3):
        rgb=np.asarray(data['train'].raw(row)[0].resize((128,128))); gray=rgb.mean(2)/255.; mask=gray>threshold_otsu(gray)
        ims=[gray,sobel(gray),canny(gray,sigma=1),mask,binary_opening(mask,disk(2)),binary_closing(mask,disk(2)),skeletonize(mask)]
        names=['gray','Sobel','Canny sigma=1','Otsu','opening disk=2','closing disk=2','skeletonize']
        for ax,z,name in zip(axes[row],ims,names): ax.imshow(z,cmap='gray'); ax.set_title(name,fontsize=9); ax.axis('off')
    fig.tight_layout(); fig.savefig(folder/'processing.png',dpi=140); plt.close(fig)

def explanations(model,ds,logits,y,device,path):
    from torch import nn
    model.eval(); good=torch.where(logits.argmax(1)==y)[0][:5].tolist(); bad=torch.where(logits.argmax(1)!=y)[0][:5].tolist(); ids=good+bad
    if not ids: return
    convs=[m for m in model.modules() if isinstance(m,nn.Conv2d)]; layer=convs[-1] if convs else None
    vit=hasattr(model,'blocks') and hasattr(model,'patch_embed')
    if vit: layer=None
    fig,axes=plt.subplots(2,5,figsize=(15,6)); capture={}; handle=None; attention=[]; handles=[]
    if vit:
        for block in model.blocks:
            block.attn.fused_attn=False
            handles.append(block.attn.attn_drop.register_forward_hook(lambda m,inp,out:attention.append(out.detach())))
    if layer:
        def hook(_,inp,out): capture['act']=out; out.retain_grad()
        handle=layer.register_forward_hook(hook)
    for ax,i in zip(axes.flat,ids):
        attention.clear(); x=ds[i][0].unsqueeze(0).to(device).requires_grad_(True); model.zero_grad(set_to_none=True)
        # Probe parameters may be frozen; input gradients still permit Grad-CAM.
        z=model(x); c=int(z.argmax(1)); z[0,c].backward()
        if vit:
            joint=None
            for a in attention:
                a=a.mean(1)[0]; a=a+torch.eye(a.shape[-1],device=a.device); a=a/a.sum(-1,keepdim=True)
                joint=a if joint is None else a@joint
            patches=joint[0,1:]; side=int(patches.numel()**.5)
            cam=torch.nn.functional.interpolate(patches.reshape(1,1,side,side),size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0]
        elif layer:
            a=capture['act']; cam=(a.grad.mean((2,3),keepdim=True)*a).sum(1).relu()[0]
            cam=torch.nn.functional.interpolate(cam[None,None],size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0]
        else: cam=x.grad.abs().max(1).values[0]
        cam=cam.detach().cpu().numpy(); cam=(cam-cam.min())/max(cam.max()-cam.min(),1e-8)
        im=ds.raw(i)[0].resize((x.shape[-1],x.shape[-2])); ax.imshow(im); ax.imshow(cam,alpha=.45,cmap='jet'); ax.axis('off')
        ax.set_title(f'True {int(y[i])} / pred {c}',fontsize=9)
    if handle: handle.remove()
    for h in handles: h.remove()
    for ax in axes.flat: ax.axis('off')
    kind='Attention rollout' if vit else 'Grad-CAM' if layer else 'Input-gradient saliency'
    fig.suptitle(f'{kind}; {len(good)} correct, {len(bad)} incorrect available'); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)

def evaluate(cfg,data,model,folder,split='val',robust=True,explain=True):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True); ds=data[split]; device='cuda' if torch.cuda.is_available() else 'cpu'; model=model.to(device)
    logits,y=predict(model,ds,device,cfg['batch']); result=metrics(logits,y,ds.classes)
    torch.save(dict(logits=logits,targets=y),folder/f'{split}_predictions.pt'); dump(folder/f'{split}_metrics.json',result)
    fig,ax=plt.subplots(figsize=(9,8)); image=ax.imshow(result['normalized_confusion'],cmap='Blues',vmin=0,vmax=1)
    ax.set_xticks(range(len(ds.classes)),ds.classes,rotation=90,fontsize=7); ax.set_yticks(range(len(ds.classes)),ds.classes,fontsize=7)
    ax.set(xlabel='Predicted class',ylabel='True class'); fig.colorbar(image,ax=ax,label='Row-normalized share')
    fig.tight_layout(); fig.savefig(folder/'confusion.png',dpi=130); plt.close(fig)
    mistakes=torch.where(logits.argmax(1)!=y)[0].tolist(); fig,axes=plt.subplots(2,5,figsize=(14,6))
    for ax,i in zip(axes.flat,mistakes[:10]): ax.imshow(ds.raw(i)[0]); ax.set_title(f'True {int(y[i])} / pred {int(logits[i].argmax())}',fontsize=9)
    for ax in axes.flat: ax.axis('off')
    fig.tight_layout(); fig.savefig(folder/'errors.png',dpi=140); plt.close(fig)
    dump(folder/'error_review.json',[dict(index=i,true=int(y[i]),predicted=int(logits[i].argmax()),cause='VISUAL_REVIEW_REQUIRED') for i in mistakes[:(len(mistakes) if cfg['variant']==5 else 20 if cfg['variant']==9 else 10)]])
    if explain: explanations(model,ds,logits,y,device,folder/'explanations.png')
    if robust:
        corruption_rows=[]; kinds=['noise:0.05','blur','bright+','bright-']
        if cfg['variant']==12: kinds+=['jpeg','noise:0.01','noise:0.03','noise:0.1','noise:0.2']
        if cfg['variant'] in [29,30]: kinds+=['shift:4','shift:8','scale:0.6','scale:1.3']
        for kind in kinds:
            z,t=predict(model,Corrupt(ds,kind,cfg['seed']),device,cfg['batch']); acc=(z.argmax(1)==t).float().mean().item()
            corruption_rows.append(dict(corruption=kind,accuracy=acc,drop=result['accuracy']-acc))
        dump(folder/'robustness.json',corruption_rows)
    rates=[latency(model,ds,device)]
    if cfg['variant']==20 and device!='cpu': rates.append(latency(model,ds,'cpu')); model.to(device)
    dump(folder/'latency.json',rates)
    coverage=[]; conf=logits.softmax(-1).max(1).values
    for threshold in np.linspace(0,1,21):
        mask=conf>=threshold
        coverage.append(dict(threshold=float(threshold),coverage=mask.float().mean().item(),accuracy=(logits.argmax(1)[mask]==y[mask]).float().mean().item() if mask.any() else None))
    dump(folder/'coverage.json',coverage)
    if cfg['dataset']=='utkface':
        groups=[ds.base.groups[i] for i in ds.indices]; rows=[]
        for group in sorted(set(groups)):
            ids=torch.tensor([i for i,g in enumerate(groups) if g==group])
            rows.append(dict(metadata_group=group,n=len(ids),accuracy=(logits[ids].argmax(1)==y[ids]).float().mean().item()))
        dump(folder/'subgroups.json',rows)
    return result
