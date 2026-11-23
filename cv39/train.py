import copy,csv,time,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from .models import build,head
from .util import seed_all,sync,dump,atomic_save,rng_state,restore_rng

def classification_loss(logits,y,mode='ce',weights=None,smoothing=0.):
    if mode=='focal':
        ce=F.cross_entropy(logits,y,reduction='none'); pt=logits.softmax(-1).gather(1,y[:,None]).squeeze(1)
        return ((1-pt).pow(2)*ce).mean()
    return F.cross_entropy(logits,y,weight=weights,label_smoothing=smoothing)

def kd_loss(student,teacher,y,T=4.,alpha=.7):
    soft=F.kl_div(F.log_softmax(student/T,-1),F.softmax(teacher/T,-1),reduction='batchmean')*T*T
    return alpha*soft+(1-alpha)*F.cross_entropy(student,y)

def mixed_batch(x,y,mixup,cutmix):
    if mixup and cutmix: raise ValueError('mixup and cutmix are separate experiments')
    if not (mixup or cutmix): return x,y,y,1.
    perm=torch.randperm(len(x),device=x.device); alpha=mixup or cutmix; lam=float(np.random.beta(alpha,alpha))
    if mixup: return lam*x+(1-lam)*x[perm],y,y[perm],lam
    _,_,h,w=x.shape; ratio=math.sqrt(1-lam); rw,rh=int(w*ratio),int(h*ratio)
    cx,cy=np.random.randint(w),np.random.randint(h)
    x1,x2=max(0,cx-rw//2),min(w,cx+rw//2); y1,y2=max(0,cy-rh//2),min(h,cy+rh//2)
    z=x.clone(); z[:,:,y1:y2,x1:x2]=x[perm,:,y1:y2,x1:x2]; lam=1-(x2-x1)*(y2-y1)/(w*h)
    return z,y,y[perm],lam

@torch.inference_mode()
def predict(model,ds,device,batch=128,tta=False):
    model.eval(); logits=[]; targets=[]
    for x,y in DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0):
        x=x.to(device); z=model(x)
        if tta: z=(z+model(x.flip(-1)))/2
        logits.append(z.float().cpu()); targets.append(y)
    return torch.cat(logits),torch.cat(targets)

def train(cfg,data,folder,initial=None,pretrained=False,phase='scratch',teacher=None,resume=True,interrupt_after=None):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True); seed_all(cfg['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=initial if initial is not None else build(cfg,len(data['train'].classes),pretrained)
    model=model.to(device); classifier=head(model); head_ids={id(p) for p in classifier.parameters()}
    if phase!='partial':
        for p in model.parameters(): p.requires_grad_(phase!='probe' or id(p) in head_ids)
    backbone=[p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]; hp=[p for p in classifier.parameters() if p.requires_grad]
    params=[{'params':backbone,'lr':1e-4},{'params':hp,'lr':1e-3}] if phase=='ft' else [{'params':[p for p in model.parameters() if p.requires_grad],'lr':cfg['lr']}]
    cls={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[cfg['optimizer']]
    kwargs=dict(weight_decay=cfg['wd'])
    if cfg['optimizer']=='sgd': kwargs['momentum']=.9
    optimizer=cls(params,**kwargs); counts=np.bincount(data['train'].targets,minlength=len(data['train'].classes))
    mode=cfg.get('loss','ce'); weights=torch.tensor(1/np.maximum(counts,1),dtype=torch.float32,device=device)
    weights=weights/weights.mean() if mode=='weighted' else None
    sampler=None
    if mode=='oversample':
        sampler=WeightedRandomSampler([1/max(counts[y],1) for y in data['train'].targets],len(data['train']),replacement=True)
    if cfg.get('sample_weights'):
        sampler=WeightedRandomSampler(cfg['sample_weights'],len(data['train']),replacement=True)
    # Global torch RNG state is saved at epoch boundaries; workers are reseeded each epoch.
    loader=DataLoader(data['train'],batch_size=cfg['batch'],shuffle=sampler is None,sampler=sampler,num_workers=cfg['workers'],persistent_workers=False)
    if cfg['scheduler']=='onecycle':
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=[g['lr'] for g in params],epochs=cfg['epochs'],steps_per_epoch=len(loader))
    elif cfg['scheduler']=='step': scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,cfg['epochs']//3),gamma=.1)
    else: scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
    amp=cfg['amp'] and device=='cuda'; scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start=0; best=-1.; bad=0; history=[]; total_seconds=0.; resumed=False
    last=folder/'last.pt'
    if resume and last.exists():
        state=torch.load(last,map_location=device,weights_only=False)
        stable=lambda c:{k:v for k,v in c.items() if k!='budget_deadline'}
        if stable(state['cfg'])!=stable(cfg): raise ValueError(f'Config changed in {folder}; use a new run directory')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start=state['epoch']+1; best=state['best']; bad=state['bad']; history=state['history']; total_seconds=state['total_seconds']; restore_rng(state['rng']); resumed=True
    if teacher:
        teacher=teacher.to(device).eval(); teacher.requires_grad_(False)
    stop_after=interrupt_after
    for epoch in range(start,cfg['epochs']):
        if cfg.get('budget_deadline') and time.time()>=cfg['budget_deadline']: break
        model.train()
        if phase=='probe': model.eval(); classifier.train()  # Freeze BN running moments as well as parameters.
        if phase=='partial':
            for module in model.modules():
                if isinstance(module,nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()): module.eval()
        total_loss=correct=n=0; sync(device); begin=time.perf_counter()
        for x,y in loader:
            x=x.to(device); y=y.to(device); original=x; x,ya,yb,lam=mixed_batch(x,y,cfg['mixup'],cfg['cutmix'])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):
                z=model(x)
                if teacher:
                    with torch.no_grad(): tz=teacher(original)
                    loss=kd_loss(z,tz,y,cfg.get('temperature',4),.7)
                else:
                    loss=lam*classification_loss(z,ya,mode,weights,cfg['smoothing'])+(1-lam)*classification_loss(z,yb,mode,weights,cfg['smoothing'])
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer); scaler.update()
            if cfg['scheduler']=='onecycle': scheduler.step()
            total_loss+=float(loss.detach())*len(y); correct+=(lam*(z.argmax(1)==ya).sum()+(1-lam)*(z.argmax(1)==yb).sum()).item(); n+=len(y)
        sync(device); seconds=time.perf_counter()-begin; total_seconds+=seconds
        logits,ys=predict(model,data['val'],device,cfg['batch']); val_acc=(logits.argmax(1)==ys).float().mean().item(); val_loss=F.cross_entropy(logits,ys).item()
        row=dict(epoch=epoch+1,train_loss=total_loss/n,train_accuracy=correct/n,val_loss=val_loss,val_accuracy=val_acc,lr=optimizer.param_groups[-1]['lr'],seconds=seconds,images_per_second=n/seconds)
        history.append(row); print(folder.name,json.dumps(row),flush=True)
        if cfg['scheduler']!='onecycle': scheduler.step()
        improved=val_acc>best; best=max(best,val_acc); bad=0 if improved else bad+1
        state=dict(cfg=cfg,classes=data['train'].classes,stats=data['stats'],model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,best=best,bad=bad,history=history,total_seconds=total_seconds,rng=rng_state(),phase=phase)
        atomic_save(last,state)
        if improved: atomic_save(folder/'best.pt',state)
        dump(folder/'history.json',history)
        with (folder/'history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        if bad>=cfg['patience'] or (stop_after and epoch+1>=stop_after): break
    if not (folder/'best.pt').exists(): raise RuntimeError('No epoch completed; budget exhausted')
    state=torch.load(folder/'best.pt',map_location=device,weights_only=False); model.load_state_dict(state['model'])
    dump(folder/'training.json',dict(phase=phase,epochs_completed=len(history),best_val_accuracy=best,train_seconds=total_seconds,resumed=resumed,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),checkpoint_bytes=(folder/'best.pt').stat().st_size,smoke=cfg['smoke']))
    return model
