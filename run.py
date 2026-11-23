"""Colab/local CLI. Run one variant at a time; state is kept in out/variant_NN."""
import argparse,copy,json,time
from pathlib import Path
import torch
from cv39.registry import config
from cv39.data import prepare,transform
from cv39.models import build
from cv39.train import train
from cv39.evaluate import evaluate,figures
from cv39.experiments import special,subset_data
from cv39.util import dump,environment,longtail

def launch(number,args):
    cfg=config(number); cfg.update(root=args.data,out=args.out,workers=args.workers)
    if args.smoke: cfg.update(epochs=2,epochs_probe=2,epochs_ft=2,train_limit=120,val_limit=48,batch=24,workers=0,smoke=True)
    folder=Path(args.out)/('smoke' if args.smoke else 'full')/f'variant_{number:02d}'; folder.mkdir(parents=True,exist_ok=True)
    dump(folder/'config.json',cfg); dump(folder/'environment.json',environment())
    dump(folder/'status.json',dict(state='running',stage=args.stage,actual_model_metrics=False))
    data=prepare(cfg,folder)
    if number==35:
        data=subset_data(data,longtail(data['train'].targets,100,cfg['seed']))
        dump(folder/'longtail_train_indices.json',data['train'].indices)
    if args.stage=='classical':
        from cv39.classical import run
        run(cfg,data,folder/'classical'); return
    if args.stage=='test':
        lock=folder/'FINAL_TEST_LOCK.json'
        if lock.exists(): raise RuntimeError('Final test already evaluated. Do not tune on test or overwrite the final lock.')
        candidates=[]
        for p in folder.rglob('training.json'):
            checkpoint=p.parent/'best.pt'
            # Accuracy of a different target-domain/rotation problem cannot select a source model.
            excluded=any(name in p.parts for name in ['adaptation','target_final','rotation_ssl'])
            if checkpoint.exists() and not excluded: candidates.append((json.loads(p.read_text())['best_val_accuracy'],checkpoint))
        if not candidates: raise RuntimeError('Train and select a model on val first')
        _,ckpt=max(candidates,key=lambda x:x[0]); state=torch.load(ckpt,map_location='cpu',weights_only=False); selected=state['cfg']
        if selected['dataset']!=cfg['dataset']: raise RuntimeError('Selected checkpoint is for another domain; select source model manually')
        d=prepare(selected,folder/'final_data'); m=build(selected,len(state['classes'])); m.load_state_dict(state['model'])
        dump(lock,dict(checkpoint=str(ckpt),state='started',note='Test access locked before evaluation'))
        result=evaluate(selected,d,m,folder/'final_test',split='test',robust=False,explain=False)
        dump(lock,dict(checkpoint=str(ckpt),state='completed',accuracy=result['accuracy'])); return
    if args.stage=='all':
        figures(data,folder)
        from cv39.classical import run
        run(cfg,data,folder/'classical')
    models={}
    if number!=39:
        for name,aug in [('none','none'),('policy',cfg['aug'])]:
            rc=dict(cfg,aug=aug)
            if name=='none': rc.update(mixup=0.,cutmix=0.)
            d=dict(data); d['train']=copy.copy(data['train']); d['train'].transform=transform(rc,True,data['stats'])
            # Deliberate epoch-boundary interruption then reload; last.pt contains optimiser, scheduler and RNG.
            train(rc,d,folder/name,interrupt_after=1)
            model=train(rc,d,folder/name)
            evaluate(rc,d,model,folder/name); models[name]=model
    else: models['policy']=None
    if args.stage in ['all','special']: special(cfg,data,models['policy'],folder/'special')
    dump(folder/'status.json',dict(state='executed_requires_review',stage=args.stage,
        actual_model_metrics=True,smoke=cfg['smoke'],test_executed=(folder/'FINAL_TEST_LOCK.json').exists(),
        remaining='Inspect images, fill REPORT_NOTES.md, verify per-variant requirements, then perform one final test.'))
    from report import build_reports
    build_reports(args.out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--variant',type=int,nargs='+',required=True); p.add_argument('--data',default='data'); p.add_argument('--out',default='runs')
    p.add_argument('--stage',choices=['all','train','classical','special','test'],default='all'); p.add_argument('--workers',type=int,default=2); p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    for number in args.variant:
        try: launch(number,args)
        except Exception as exc:
            cfg=config(number); folder=Path(args.out)/('smoke' if args.smoke else 'full')/f'variant_{number:02d}'
            dump(folder/'status.json',dict(state='failed',error=type(exc).__name__,detail=str(exc),stage=args.stage))
            raise
if __name__=='__main__': main()
