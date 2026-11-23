import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self,classes,width=64,flatten=False):
        super().__init__(); layers=[]; inc=3
        for out in [width,width*2,width*4]:
            layers += [nn.Conv2d(inc,out,3,padding=1),nn.BatchNorm2d(out),nn.ReLU(),
                       nn.Conv2d(out,out,3,padding=1),nn.BatchNorm2d(out),nn.ReLU(),nn.MaxPool2d(2)]
            inc=out
        self.features=nn.Sequential(*layers); self.pool=nn.AdaptiveAvgPool2d((4,4) if flatten else 1)
        self.head=nn.Linear(inc*(16 if flatten else 1),classes)
    def forward(self,x): return self.head(self.pool(self.features(x)).flatten(1))

class LeNet(nn.Module):
    def __init__(self,classes):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(3,6,5,padding=2),nn.Tanh(),nn.AvgPool2d(2),
            nn.Conv2d(6,16,5),nn.Tanh(),nn.AvgPool2d(2),nn.AdaptiveAvgPool2d((5,5)))
        self.head=nn.Sequential(nn.Linear(400,120),nn.Tanh(),nn.Linear(120,84),nn.Tanh(),nn.Linear(84,classes))
    def forward(self,x): return self.head(self.features(x).flatten(1))

class MLP(nn.Module):
    def __init__(self,classes,size,width=80):
        super().__init__(); self.head=nn.Sequential(nn.Flatten(),nn.Linear(3*size*size,width),nn.ReLU(),
            nn.Linear(width,width),nn.ReLU(),nn.Linear(width,classes))
    def forward(self,x): return self.head(x)

class Block(nn.Module):
    def __init__(self,inc,out,stride):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(inc,out,3,stride,1,bias=False),nn.BatchNorm2d(out),nn.ReLU(),
                                                nn.Conv2d(out,out,3,1,1,bias=False),nn.BatchNorm2d(out))
        self.skip=nn.Identity() if inc==out and stride==1 else nn.Sequential(nn.Conv2d(inc,out,1,stride,bias=False),nn.BatchNorm2d(out))
    def forward(self,x): return torch.relu(self.net(x)+self.skip(x))

class ResNet20(nn.Module):
    def __init__(self,classes):
        super().__init__(); layers=[nn.Conv2d(3,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.ReLU()]; inc=16
        for group,out in enumerate([16,32,64]):
            for b in range(3):
                stride=2 if group>0 and b==0 else 1; layers.append(Block(inc,out,stride)); inc=out
        self.features=nn.Sequential(*layers); self.head=nn.Linear(64,classes)
    def forward(self,x): return self.head(self.features(x).mean((2,3)))

def head(model):
    if hasattr(model,'fc'): return model.fc
    if hasattr(model,'classifier'): return model.classifier
    return model.head

def set_head(model,new):
    if hasattr(model,'fc'): model.fc=new
    elif hasattr(model,'classifier'): model.classifier=new
    else: model.head=new

def build(cfg,classes,pretrained=False):
    from torchvision import models
    name=cfg['model']; size=cfg['size']
    if name=='smallcnn': model=SmallCNN(classes,flatten=cfg.get('flatten',False))
    elif name=='lenet': model=LeNet(classes)
    elif name=='mlp':
        # Match actual parameter count to reference LeNet/SmallCNN as requested.
        reference=SmallCNN(classes) if cfg['variant']==30 else LeNet(classes)
        target=sum(p.numel() for p in reference.parameters())
        widths=range(1,1025); w=min(widths,key=lambda w:abs((3*size*size*w+w+w*w+w+w*classes+classes)-target))
        model=MLP(classes,size,w)
    elif name=='resnet20': model=ResNet20(classes)
    elif name=='vit_tiny':
        import timm
        model=timm.create_model('vit_tiny_patch16_224',pretrained=pretrained,num_classes=classes)
        for block in model.blocks: block.attn.fused_attn=False
    else:
        keys={'resnet18':(models.resnet18,models.ResNet18_Weights.DEFAULT),
              'resnet50':(models.resnet50,models.ResNet50_Weights.DEFAULT),
              'mobilenet_v3':(models.mobilenet_v3_small,models.MobileNet_V3_Small_Weights.DEFAULT),
              'densenet121':(models.densenet121,models.DenseNet121_Weights.DEFAULT),
              'efficientnet_b0':(models.efficientnet_b0,models.EfficientNet_B0_Weights.DEFAULT)}
        factory,weights=keys[name]; model=factory(weights=weights if pretrained else None)
        old=head(model)
        if isinstance(old,nn.Linear): set_head(model,nn.Linear(old.in_features,classes))
        else:
            old[-1]=nn.Linear(old[-1].in_features,classes)
        if name=='resnet18' and not pretrained and size<=64:
            model.conv1=nn.Conv2d(3,64,3,1,1,bias=False); model.maxpool=nn.Identity()
    if cfg.get('groupnorm'):
        def replace(parent):
            for key,child in list(parent.named_children()):
                if isinstance(child,nn.BatchNorm2d):
                    groups=next(g for g in [32,16,8,4,2,1] if child.num_features%g==0)
                    setattr(parent,key,nn.GroupNorm(groups,child.num_features))
                else: replace(child)
        replace(model)
    if cfg.get('dropout',0): set_head(model,nn.Sequential(nn.Dropout(cfg['dropout']),head(model)))
    return model
