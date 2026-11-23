"""Variant-specific experiments. Artifacts are emitted only after actual execution."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from . import data as datasets
from .data import prepare,Wrapped,transform,Shapes
from .models import build,head,set_head
from .train import train,predict
from .evaluate import evaluate,metrics,latency,Corrupt
from .util import dump,seed_all,fewshot,longtail

def subset_data(data,ids):
    result=dict(data); result['train']=copy.copy(data['train']); result['train'].indices=[data['train'].indices[i] for i in ids]
    result['train'].targets=[data['train'].targets[i] for i in ids]; return result

def transfer(cfg,folder,model_name=None,initial=None):
    cfg=copy.deepcopy(cfg); cfg.update(model=model_name or cfg['model'],size=224,norm='imagenet',lr=1e-3)
    if cfg['model'] in ['smallcnn','lenet','mlp','resnet20']: cfg['model']='resnet18'
    data=prepare(cfg,Path(folder)/'data'); base_cfg=copy.deepcopy(cfg)
    pc=copy.deepcopy(base_cfg); pc['epochs']=cfg['epochs_probe']
    probe=train(pc,data,Path(folder)/'probe',initial=initial,pretrained=initial is None,phase='probe')
    evaluate(pc,data,probe,Path(folder)/'probe')
    fc=copy.deepcopy(base_cfg); fc['epochs']=cfg['epochs_ft']
    ft=train(fc,data,Path(folder)/'finetune',initial=probe,phase='ft')
    evaluate(fc,data,ft,Path(folder)/'finetune')
    return ft,data,cfg

def repeated_fewshot(cfg,data,folder,k,transfer_mode=False):
    results=[]
    # Same k-subset across repetitions; only initialisation / training seed changes.
    subset=fewshot(data['train'].targets,k,cfg['seed'])
    for repeat in range(3):
        rc=copy.deepcopy(cfg); rc['seed']=cfg['seed']+100*repeat
        d=subset_data(data,subset); name=Path(folder)/f'k{k}_repeat{repeat}'
        if transfer_mode:
            pc=dict(rc,epochs=cfg['epochs_probe'],lr=1e-3)
            model=train(pc,d,name/'probe',pretrained=True,phase='probe')
            model=train(dict(rc,epochs=cfg['epochs_ft']),d,name/'ft',initial=model,phase='ft')
        else: model=train(rc,d,name)
        z,y=predict(model,d['val'],'cuda' if torch.cuda.is_available() else 'cpu',rc['batch'])
        m=metrics(z,y,d['val'].classes); results.append(dict(repeat=repeat,accuracy=m['accuracy'],macro_F1=m['macro_F1']))
    dump(Path(folder)/f'fewshot_k{k}.json',dict(k=k,subset_indices=subset,results=results,
        mean=float(np.mean([r['accuracy'] for r in results])),std=float(np.std([r['accuracy'] for r in results],ddof=1))))

class TeacherInput(nn.Module):
    def __init__(self,model,stats):
        super().__init__(); self.model=model; self.register_buffer('mean',torch.tensor(stats[0])[None,:,None,None]); self.register_buffer('std',torch.tensor(stats[1])[None,:,None,None])
    def forward(self,x):
        raw=(x*self.std+self.mean).clamp(0,1); raw=torch.nn.functional.interpolate(raw,(224,224),mode='bilinear',align_corners=False)
        mean=raw.new_tensor([.485,.456,.406])[None,:,None,None]; std=raw.new_tensor([.229,.224,.225])[None,:,None,None]
        return self.model((raw-mean)/std)

class RotationDataset(Dataset):
    def __init__(self,base): self.base=base; self.classes=['0','90','180','270']; self.targets=[i%4 for i in range(len(base))]
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        x,_=self.base[i]; k=i%4; return torch.rot90(x,k,(-2,-1)),k

def special(cfg,data,base,folder):
    folder=Path(folder); exp=cfg['experiment']; device='cuda' if torch.cuda.is_available() else 'cpu'
    if exp in ['transfer','unfreeze']:
        model,d,tc=transfer(cfg,folder/'transfer')
        if exp=='unfreeze':
            # Compare freeze fractions as independent runs from the same ImageNet checkpoint.
            for fraction in [0.,.5,1.]:
                seed_all(cfg['seed']); m=build(tc,len(d['train'].classes),True)
                blocks=[m.layer1,m.layer2,m.layer3,m.layer4]
                if fraction>0:
                    m.conv1.requires_grad_(False); m.bn1.requires_grad_(False)
                for block in blocks[:int(4*fraction)]: block.requires_grad_(False)
                # Engine normally sets all requires_grad; preserve frozen block mask below.
                sc=dict(tc,epochs=cfg['epochs_ft'],freeze_fraction=fraction)
                train(sc,d,folder/f'freeze_{fraction}',initial=m,phase='partial')
    elif exp.startswith('fewshot'):
        k=int(exp.split(':')[1])
        if cfg['transfer']:
            model,d,tc=transfer(cfg,folder/'transfer')
            for value in sorted(set([5,10,25,k,100])) if cfg['variant']==13 else [k]: repeated_fewshot(tc,d,folder/'fewshot',value,True)
        else: repeated_fewshot(cfg,data,folder/'fewshot',k)
    elif exp in ['imbalance','longtail','balanced_ft']:
        if exp=='balanced_ft':
            model,d,tc=transfer(cfg,folder/'transfer')
            # For age/subgroup rebalancing use joint label+metadata, not predicted attributes.
            labels=[]
            for idx,y in zip(d['train'].indices,d['train'].targets): labels.append((y,d['train'].base.groups[idx]))
            counts={x:labels.count(x) for x in set(labels)}
            weights=[1/counts[x] for x in labels]
            rc=dict(tc,epochs=cfg['epochs_ft'],sample_weights=weights)
            balanced=train(rc,d,folder/'balanced_ft',initial=copy.deepcopy(model),phase='ft'); evaluate(rc,d,balanced,folder/'balanced_ft')
        else:
            for loss in ['weighted','oversample','focal']:
                rc=dict(cfg,loss=loss); m=train(rc,data,folder/loss); evaluate(rc,data,m,folder/loss)
    elif exp=='robust_ft':
        rc=dict(cfg,aug='corruption',epochs=cfg['epochs_ft'],lr=1e-3)
        d=dict(data); d['train']=copy.copy(data['train']); d['train'].transform=transform(rc,True,data['stats'])
        model=train(rc,d,folder/'robust_ft',initial=copy.deepcopy(base)); evaluate(rc,d,model,folder/'robust_ft')
    elif exp in ['augmentation_sweep','optimizer_sweep','scheduler_sweep','regularization']:
        if exp=='augmentation_sweep': settings=[dict(aug=a,cutmix=c) for a,c in [('none',0.),('basic',0.),('strong',0.),('strong',1.)]]
        elif exp=='optimizer_sweep': settings=[dict(optimizer=o,lr=lr) for o,lrs in [('sgd',[.03,.1]),('adam',[3e-4,1e-3]),('adamw',[3e-4,1e-3])] for lr in lrs]
        elif exp=='scheduler_sweep': settings=[dict(scheduler=s) for s in ['step','cosine','onecycle']]
        else: settings=[dict(),dict(dropout=.3),dict(smoothing=.1),dict(groupnorm=True)]
        rows=[]
        for i,setting in enumerate(settings):
            rc=dict(cfg,**setting); d=dict(data); d['train']=copy.copy(data['train']); d['train'].transform=transform(rc,True,data['stats'])
            m=train(rc,d,folder/f'sweep_{i}'); met=evaluate(rc,d,m,folder/f'sweep_{i}')
            rows.append(dict(setting=setting,accuracy=met['accuracy'],macro_F1=met['macro_F1'],run=f'sweep_{i}')); dump(folder/'sweep.json',rows)
        best=max(rows,key=lambda r:r['accuracy']); tc=dict(cfg,**best['setting'])
        transfer(tc,folder/'best_transfer','resnet18')
    elif exp=='distillation':
        teacher,_,_=transfer(dict(cfg,model='resnet50'),folder/'teacher','resnet50')
        # Sweep includes the mandatory T=4 and both bonus temperatures.
        for temp in [2,4,8]:
            rc=dict(cfg,temperature=temp); model=train(rc,data,folder/f'kd_T{temp}',teacher=TeacherInput(copy.deepcopy(teacher),data['stats']))
            evaluate(rc,data,model,folder/f'kd_T{temp}')
    elif exp=='compression':
        import torch_pruning as tp
        from torch.ao.quantization.quantize_fx import prepare_fx,convert_fx
        from torch.ao.quantization import get_default_qconfig_mapping
        base=base.cpu().eval(); example=data['train'][0][0][None]; results=[]
        for pruned in [False,True]:
            model=copy.deepcopy(base)
            if pruned:
                # Physical channel removal with dependency propagation, not zero masks.
                model.requires_grad_(True)
                pruner=tp.pruner.MagnitudePruner(model,example_inputs=example,importance=tp.importance.MagnitudeImportance(p=2),pruning_ratio=.5,ignored_layers=[head(model)])
                pruner.step(); torch.save(model,folder/'physical_pruned_model.pt')
            for quantized in [False,True]:
                name=f'prune{int(pruned)}_int8{int(quantized)}'; m=copy.deepcopy(model).eval()
                if quantized:
                    backend='x86' if 'x86' in torch.backends.quantized.supported_engines else 'fbgemm'; torch.backends.quantized.engine=backend
                    prepared=prepare_fx(m,get_default_qconfig_mapping(backend),(example,))
                    with torch.inference_mode():
                        for i in range(min(128,len(data['train']))): prepared(data['train'][i][0][None])
                    m=convert_fx(prepared)
                path=folder/(name+'.pt'); torch.save(m.state_dict(),path)
                z,y=predict(m,data['val'],'cpu',cfg['batch']); met=metrics(z,y,data['val'].classes)
                results.append(dict(name=name,accuracy=met['accuracy'],macro_F1=met['macro_F1'],bytes=path.stat().st_size,parameters=sum(p.numel() for p in m.parameters()),latency=latency(m,data['val'],'cpu')))
                dump(folder/'compression.json',results)
    elif exp=='ssl':
        n=len(data['train']); order=np.random.default_rng(cfg['seed']).permutation(n); boundary=int(.7*n)
        ssl_d=subset_data(data,order[:boundary]); probe_pool=subset_data(data,order[boundary:])
        rot=dict(ssl_d); rot['train']=RotationDataset(ssl_d['train']); rot['val']=RotationDataset(data['val']); rot['stats']=data['stats']
        sc=dict(cfg,model='resnet18'); backbone=train(sc,rot,folder/'rotation_ssl')
        dump(folder/'ssl_partition.json',dict(unlabeled_indices=order[:boundary].tolist(),probe_pool_indices=order[boundary:].tolist()))
        for fraction in [.1,.5,1.]:
            d=subset_data(probe_pool,np.arange(max(1,int(len(probe_pool['train'])*fraction))))
            m=build(cfg,len(data['train'].classes)); state=backbone.state_dict(); own=m.state_dict()
            m.load_state_dict({k:v for k,v in state.items() if k in own and v.shape==own[k].shape and not k.startswith('fc.')},strict=False)
            pc=dict(cfg,epochs=cfg['epochs_probe'],lr=1e-3)
            model=train(pc,d,folder/f'probe_{fraction}',initial=m,phase='probe'); evaluate(pc,d,model,folder/f'probe_{fraction}')
            model=train(cfg,d,folder/f'supervised_{fraction}'); evaluate(cfg,d,model,folder/f'supervised_{fraction}')
    elif exp in ['curriculum','invariance']:
        # Same label ontology; actual novel-color and spatial shifts can be scored directly.
        target=Shapes(1200,cfg['data_seed']+200000,domain='newcolor')
        td=dict(data); td['val']=Wrapped(target,range(len(target)),{0:0,1:1,2:2},data['val'].transform)
        evaluate(cfg,td,base,folder/'new_color',explain=False)
        if exp=='curriculum':
            easy=Shapes(6000,cfg['data_seed'],domain='easy'); d=dict(data); d['train']=Wrapped(easy,data['train'].indices,{0:0,1:1,2:2},data['train'].transform)
            rc=dict(cfg,epochs=max(1,cfg['epochs']//2)); m=train(rc,d,folder/'easy'); m=train(rc,data,folder/'hard',initial=m); evaluate(cfg,td,m,folder/'curriculum_novel')
    elif exp.startswith('domain:') or exp=='texture_shift':
        if cfg['transfer']:
            base,data,cfg=transfer(cfg,folder/'source_transfer')
        if exp=='texture_shift':
            flat_cfg=dict(cfg,flatten=True)
            flat=train(flat_cfg,data,folder/'flatten_ablation'); evaluate(flat_cfg,data,flat,folder/'flatten_ablation')
        target_name=exp.split(':')[1] if ':' in exp else 'patterns'; tc=dict(cfg,dataset=target_name,subset=None,transfer=False)
        if target_name=='emnist': tc['size']=cfg['size']
        target=prepare(tc,folder/'target_data'); m=build(cfg,len(target['train'].classes)); own=m.state_dict()
        # Preserve source feature extractor, reinitialise classifier for different label spaces.
        state={k:v for k,v in base.state_dict().items() if k in own and v.shape==own[k].shape and not any(k.startswith(prefix) for prefix in ['head.','fc.','classifier.'])}
        m.load_state_dict(state,strict=False); source_head=copy.deepcopy(head(base)); same_semantics={cfg['dataset'],target_name}=={'svhn','mnist'}
        if same_semantics: set_head(m,copy.deepcopy(source_head))
        dump(folder/'domain_protocol.json',dict(source=cfg['dataset'],target=target_name,shared_semantic_labels=same_semantics,
             before_adaptation_accuracy_meaningful=same_semantics,head_reinitialised=not same_semantics,
             warning='MNIST vs KMNIST/EMNIST and EuroSAT vs UCM do not have interchangeable numeric class IDs.'))
        m=m.to(device)
        if same_semantics:
            z,y=predict(m,target['val'],device,cfg['batch']); dump(folder/'before_adaptation.json',metrics(z,y,target['val'].classes))
        trajectory=[]
        # One training epoch at a time via checkpoint interruption, with target/source evaluation.
        rc=dict(tc,epochs=cfg['epochs_ft'],lr=1e-3)
        for epoch in range(1,rc['epochs']+1):
            m=train(rc,target,folder/'adaptation',initial=m,interrupt_after=epoch)
            tz,ty=predict(m,target['val'],device,cfg['batch']); target_acc=(tz.argmax(1)==ty).float().mean().item()
            target_head=head(m); set_head(m,copy.deepcopy(source_head).to(device))
            sz,sy=predict(m,data['val'],device,cfg['batch']); set_head(m,target_head)
            trajectory.append(dict(epoch=epoch,target_accuracy=target_acc,source_accuracy=(sz.argmax(1)==sy).float().mean().item()))
            dump(folder/'adaptation_curve.json',trajectory)
        evaluate(rc,target,m,folder/'target_final')
    elif exp=='capstone':
        # All candidate runs share the total 30-minute training budget.
        settings=[dict(aug='none',mixup=0.,amp=False),dict(aug='strong',mixup=0.,amp=False),
                  dict(aug='strong',mixup=1.,amp=False),dict(aug='strong',mixup=1.,amp=True)]
        models=[]; rows=[]; budget=1800.
        for i,setting in enumerate(settings):
            consumed=sum(json.loads(p.read_text())['train_seconds'] for p in folder.rglob('training.json'))
            if consumed>=budget: break
            rc=dict(cfg,**setting,budget_deadline=time.time()+budget-consumed)
            m,d,tc=transfer(rc,folder/f'component_{i}','efficientnet_b0'); models.append(m)
            z,y=predict(m,d['val'],device,tc['batch']); mz=metrics(z,y,d['val'].classes)
            ztta,_=predict(m,d['val'],device,tc['batch'],tta=True)
            rows.append(dict(component=i,settings=setting,accuracy=mz['accuracy'],TTA_accuracy=metrics(ztta,y,d['val'].classes)['accuracy']))
            if len(models)>=2:
                previous,_=predict(models[-2],d['val'],device,tc['batch'],tta=True)
                rows[-1]['ensemble_accuracy']=metrics((previous+ztta)/2,y,d['val'].classes)['accuracy']
            dump(folder/'components.json',rows)
        dump(folder/'budget.json',dict(training_budget_seconds=budget,executed_components=len(rows),requested_components=len(settings),complete=len(rows)==len(settings)))
    else: raise ValueError(exp)
