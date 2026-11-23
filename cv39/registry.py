"""Explicit configuration registry; seeds for paired data are shared deliberately."""
from copy import deepcopy

# dataset, architecture, augmentation, epochs, stage-8 experiment, classical baseline(s)
ROWS = [
('cifar10','smallcnn','basic',30,'transfer',['hog']),
('cifar10','resnet18','basic',30,'unfreeze',['hog','raw']),
('cifar10','vit_tiny','strong',40,'transfer',['siftbow']),
('cifar100','resnet20','basic',30,'transfer',['hog:knn']),
('mnist','lenet','rotate10',15,'domain:kmnist',['raw:knn','hog:knn']),
('mnist','mlp','noise',15,'fewshot:10',['raw']),
('fashion_mnist','smallcnn','basic',30,'domain:mnist',['hog','lbp']),
('fashion_mnist','resnet18','basic',30,'transfer',['lbp']),
('kmnist','smallcnn','rotate15',30,'domain:emnist',['hog:knn']),
('emnist','lenet','rotate10',20,'imbalance',['hog']),
('svhn','resnet20','digit_crop',30,'domain:mnist',['colorhist','hog']),
('svhn','smallcnn','corruption',30,'robust_ft',['cannyhog']),
('stl10','resnet18','basic',30,'fewshot:50',['siftbow']),
('flowers102','resnet18','strong',30,'transfer',['siftbow']),
('flowers102','vit_tiny','strong',30,'transfer',['colorhist+hog']),
('dtd','smallcnn','texture',30,'texture_shift',['lbp']),
('eurosat','resnet18','basic',30,'domain:ucmerced',['colorhist+hog']),
('eurosat','smallcnn','basic',30,'transfer',['hog']),
('ucmerced','resnet18','basic',30,'domain:eurosat',['hog+colorhist']),
('oxford_pets','mobilenet_v3','basic',30,'fewshot:10',['hog']),
('oxford_pets','densenet121','basic',30,'transfer',['siftbow']),
('food101','efficientnet_b0','basic',30,'transfer',['colorhist+lbp']),
('food101','resnet18','basic',30,'transfer',['lbp']),
('caltech101','resnet18','basic',30,'transfer',['siftbow']),
('quickdraw','smallcnn','rotate20',30,'domain:emnist',['raw:knn','skeletonhog']),
('gtsrb','resnet20','sign_crop',30,'robust_ft',['hog+maskhist']),
('gtsrb','mobilenet_v3','sign_crop',30,'transfer',['maskedhog']),
('utkface','resnet18','basic',30,'balanced_ft',['hog']),
('synthetic','smallcnn','basic',20,'curriculum',['hu:knn']),
('synthetic','mlp','basic',20,'invariance',['hu:knn','raw:knn']),
('cifar10','smallcnn','basic',30,'augmentation_sweep',['hog']),
('cifar10','resnet18','basic',30,'optimizer_sweep',['hog']),
('cifar10','resnet18','basic',30,'scheduler_sweep',['hog']),
('cifar10','resnet18','basic',30,'regularization',['hog']),
('cifar10','resnet18','basic',30,'longtail',['hog']),
('cifar10','smallcnn','basic',30,'distillation',['hog']),
('cifar10','resnet18','basic',30,'compression',['hog']),
('cifar10','resnet18','strong',30,'ssl',['hog']),
('cifar10','efficientnet_b0','strong',30,'capstone',['hog+siftbow+colorhist'])]

PAIRED_SEEDS={15:1014,18:1017,19:1017,21:1020,23:1022,30:1029}
TRANSFER_VARIANTS={8,13,14,15,17,19,20,21,22,23,24,27,28,39}

def config(number):
    if not 1<=number<=39: raise ValueError('Variant must be 1..39')
    dataset,model,aug,epochs,experiment,features=ROWS[number-1]
    cfg=dict(variant=number,seed=1000+number,data_seed=PAIRED_SEEDS.get(number,1000+number),
        dataset=dataset,model=model,aug=aug,epochs=epochs,experiment=experiment,classical=features,
        batch=128,optimizer='sgd',lr=.1,scheduler='cosine',wd=5e-4,patience=10,
        transfer=number in TRANSFER_VARIANTS,size=32,norm='dataset',mixup=0.,cutmix=0.,
        smoothing=0.,dropout=0.,groupnorm=False,amp=True,workers=2,subset=None,
        train_limit=None,val_limit=None,smoke=False,epochs_ft=10,epochs_probe=10,
        rate_usd_hour=.2,price_is_assumption=True,root='data',out='runs',budget_seconds=None)
    cfg['size']={'mnist':28,'fashion_mnist':28,'kmnist':28,'emnist':28,'stl10':96,'dtd':128,
                 'eurosat':64,'gtsrb':48,'synthetic':32}.get(dataset,224 if cfg['transfer'] else 32)
    if model=='vit_tiny': cfg.update(size=224,optimizer='adamw',lr=3e-4)
    if dataset in ['flowers102','food101','caltech101']: cfg['subset']={'flowers102':20,'food101':10,'caltech101':10}[dataset]
    if number==22: cfg['mixup']=1.
    if number==23: cfg['cutmix']=1.
    if number==39: cfg.update(optimizer='adamw',lr=3e-4,mixup=1.,smoothing=.1,budget_seconds=1800)
    return deepcopy(cfg)
