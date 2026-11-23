import copy,json,tempfile,unittest
import contextlib,uuid,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cv39.registry import config,ROWS
from cv39.data import Shapes,Wrapped,transform,Patterns,prepare
from cv39.models import build,head,LeNet
from cv39.train import train,kd_loss,mixed_batch,classification_loss
from cv39.util import fewshot,longtail,seed_all
from cv39.evaluate import metrics,Corrupt
from poetry.train_poetry import prepare as prepare_poems,collator

torch.set_num_threads(2)

@contextlib.contextmanager
def working_directory():
    root=(Path(__file__).resolve().parent/'scratch').resolve(); root.mkdir(exist_ok=True)
    path=root/uuid.uuid4().hex; path.mkdir()
    try: yield str(path)
    finally:
        if path.resolve().parent!=root: raise RuntimeError('Refuse unsafe cleanup')
        shutil.rmtree(path)

class CoreTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(len(ROWS),39)
        for i in range(1,40): self.assertEqual(config(i)['seed'],1000+i)
    def test_paired_subsets(self):
        for a,b in [(14,15),(17,18),(20,21),(22,23),(29,30)]: self.assertEqual(config(a)['data_seed'],config(b)['data_seed'])
    def test_no_digit_flips(self):
        for i in [5,6,9,10,11,12,25,26,27]: self.assertNotIn(config(i)['aug'],['basic','strong'])
    def test_shapes_reproducible_and_independent(self):
        a=Shapes(30,12); b=Shapes(30,12); c=Shapes(30,13)
        self.assertTrue(np.array_equal(a[5][0],b[5][0])); self.assertFalse(np.array_equal(a[5][0],c[5][0]))
    def test_new_color(self):
        im,_=Shapes(30,42,'newcolor')[0]; self.assertIn((220,190,50),set(map(tuple,np.asarray(im).reshape(-1,3))))
    def test_patterns(self): self.assertEqual(Patterns(12)[8][0].size,(128,128))
    def test_prepare_persists_split(self):
        cfg=config(29); cfg.update(train_limit=48,val_limit=24)
        with working_directory() as folder:
            data=prepare(cfg,folder); m=json.loads((Path(folder)/'split.json').read_text())
            self.assertEqual(len(data['train']),48); self.assertEqual(len(m['val']),24)
            self.assertFalse(set(m['train'])&set(m['val'])); self.assertEqual(m['selected_classes'],[0,1,2])
    def test_model_shapes(self):
        for name in ['lenet','mlp','smallcnn','resnet20','resnet18']:
            cfg=config(29); cfg['model']=name
            m=build(cfg,3).eval(); self.assertEqual(m(torch.rand(2,3,32,32)).shape,(2,3))
    def test_mlp_parameter_match(self):
        cfg=config(6); m=build(cfg,10); a=sum(p.numel() for p in m.parameters()); b=sum(p.numel() for p in LeNet(10).parameters())
        self.assertLess(abs(a-b)/b,.03)
    def test_groupnorm(self):
        cfg=config(34); cfg['groupnorm']=True; m=build(cfg,10)
        self.assertFalse(any(isinstance(x,torch.nn.BatchNorm2d) for x in m.modules()))
    def test_fewshot(self):
        y=[0]*20+[1]*20; ids=fewshot(y,5,3); self.assertEqual(len(ids),10); self.assertEqual(sum(y[i]==0 for i in ids),5)
        self.assertEqual(ids,fewshot(y,5,3))
    def test_longtail(self):
        y=np.repeat(np.arange(10),100); ids=longtail(y,100,4); counts=np.bincount(y[ids]); self.assertEqual(counts[0],100); self.assertEqual(counts[-1],1)
    def test_kd_gradients(self):
        s=torch.randn(5,3,requires_grad=True); t=torch.randn(5,3); loss=kd_loss(s,t,torch.arange(5)%3); loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all())
    def test_focal(self):
        z=torch.tensor([[10.,-10.]],requires_grad=True); y=torch.tensor([0]); l=classification_loss(z,y,'focal'); self.assertLess(l.item(),1e-5)
    def test_mixup_bounds(self):
        x=torch.rand(8,3,16,16); y=torch.arange(8); z,a,b,lam=mixed_batch(x,y,1.,0.)
        self.assertTrue(0<=lam<=1); self.assertTrue((z>=0).all() and (z<=1).all()); self.assertEqual(z.shape,x.shape)
    def test_cutmix_area(self):
        seed_all(2); x=torch.arange(8.)[:,None,None,None].expand(8,3,16,16).clone(); y=torch.arange(8)
        z,a,b,lam=mixed_batch(x,y,0.,1.); self.assertTrue(0<=lam<=1)
        changed=(z[0,0]!=x[0,0]).float().mean().item()
        if b[0]!=0: self.assertAlmostEqual(changed,1-lam)
    def test_metric_orientation(self):
        logits=torch.tensor([[8.,1.,0.],[7.,3.,1.],[0.,1.,8.]]); y=torch.tensor([0,1,2]); m=metrics(logits,y,['a','b','c'])
        self.assertAlmostEqual(m['accuracy'],2/3,places=6); self.assertEqual(m['confusion'][1][0],1); self.assertEqual(m['top3'],1.)
    def test_corruption_deterministic(self):
        cfg=config(29); ds=Wrapped(Shapes(12),range(12),{0:0,1:1,2:2},transform(cfg,False,([0,0,0],[1,1,1])))
        self.assertTrue(torch.equal(Corrupt(ds,'noise:0.05',3)[5][0],Corrupt(ds,'noise:0.05',3)[5][0]))
    def test_poetry_split_no_duplicate_leakage(self):
        with working_directory() as folder:
            p=Path(folder)/'poems.tsv'; rows=['text\tpoet']+[f'poem {i} '+'word '*12+'\twriter' for i in range(50)]; p.write_text('\n'.join(rows+rows[1:10]),encoding='utf-8')
            tr,va,_=prepare_poems(p,'text','poet'); self.assertEqual(len(tr)+len(va),50); self.assertFalse({r['id'] for r in tr}&{r['id'] for r in va})
    def test_padding_does_not_mask_real_eos(self):
        class Tokenizer:
            def pad(self,batch,return_tensors):
                return dict(input_ids=torch.tensor([[7,2,2],[8,9,2]]),attention_mask=torch.tensor([[1,1,0],[1,1,1]]))
        out=collator(Tokenizer())([{},{}])
        self.assertEqual(out['labels'].tolist(),[[7,2,-100],[8,9,2]])
    def test_resume_matches_uninterrupted(self):
        cfg=config(29); cfg.update(epochs=2,batch=8,workers=0,amp=False,aug='noise',patience=10,smoke=True)
        stats=([.2]*3,[.3]*3); base=Shapes(48,10)
        ds={'train':Wrapped(base,range(32),{0:0,1:1,2:2},transform(cfg,True,stats)),
            'val':Wrapped(base,range(32,48),{0:0,1:1,2:2},transform(cfg,False,stats)),'stats':stats}
        with working_directory() as folder:
            full=Path(folder)/'full'; resumed=Path(folder)/'resumed'
            train(cfg,ds,full); train(cfg,ds,resumed,interrupt_after=1); train(cfg,ds,resumed)
            a=torch.load(full/'last.pt',weights_only=False,map_location='cpu'); b=torch.load(resumed/'last.pt',weights_only=False,map_location='cpu')
            self.assertEqual(a['epoch'],b['epoch'])
            for key in a['model']: self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
    def test_probe_freezes_bn(self):
        cfg=config(29); cfg.update(epochs=1,batch=8,workers=0,amp=False,smoke=True)
        stats=([0]*3,[1]*3); ds0=Shapes(24,99)
        d={'train':Wrapped(ds0,range(16),{0:0,1:1,2:2},transform(cfg,True,stats)),
           'val':Wrapped(ds0,range(16,24),{0:0,1:1,2:2},transform(cfg,False,stats)),'stats':stats}
        seed_all(1); m=build(cfg,3); before={k:v.clone() for k,v in m.state_dict().items() if 'running_' in k}
        with working_directory() as folder:
            train(dict(cfg,lr=1e-3),d,folder,initial=m,phase='probe')
            for k,v in before.items(): self.assertTrue(torch.equal(v,m.state_dict()[k]))

if __name__=='__main__': unittest.main(verbosity=2)
