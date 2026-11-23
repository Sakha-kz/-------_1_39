"""Hand-crafted features. Fit vocabulary and scaler on train only."""
import time,pickle
from pathlib import Path
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report
from .util import dump

def image_array(im): return np.asarray(im.convert('RGB').resize((64,64)))
def feature(im,kind,bow=None):
    from skimage.feature import hog,local_binary_pattern,canny
    from skimage.morphology import skeletonize
    rgb=image_array(im); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    if '+' in kind: return np.concatenate([feature(im,k,bow) for k in kind.split('+')])
    if kind=='raw': return cv2.resize(gray,(16,16)).ravel().astype(float)/255
    if kind in ['hog','cannyhog','skeletonhog','maskedhog']:
        source=gray
        if kind=='cannyhog': source=canny(gray/255.,sigma=1).astype(float)
        if kind=='skeletonhog': source=skeletonize(gray>cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]).astype(float)
        if kind=='maskedhog':
            mask=(((hsv[:,:,0]<10)|(hsv[:,:,0]>170)|((hsv[:,:,0]>90)&(hsv[:,:,0]<130)))&(hsv[:,:,1]>80))
            source=gray*mask
        return hog(source,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2))
    if kind=='lbp':
        x=local_binary_pattern(gray,8,1,method='uniform'); h=np.histogram(x,bins=10,range=(0,10))[0].astype(float); return h/max(h.sum(),1)
    if kind=='colorhist':
        h=cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256]).ravel(); return h/max(h.sum(),1)
    if kind=='maskhist':
        return np.array([np.mean(((hsv[:,:,0]<10)|(hsv[:,:,0]>170))&(hsv[:,:,1]>80)),np.mean((hsv[:,:,0]>90)&(hsv[:,:,0]<130)&(hsv[:,:,1]>80))])
    if kind=='hu':
        _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        hu=cv2.HuMoments(cv2.moments(mask,binaryImage=True)).ravel(); return -np.sign(hu)*np.log10(np.abs(hu)+1e-30)
    if kind=='siftbow':
        if bow is None: raise ValueError('SIFT vocabulary must be trained first')
        _,desc=cv2.SIFT_create().detectAndCompute(gray,None)
        h=np.bincount(bow.predict(desc),minlength=64).astype(float) if desc is not None else np.zeros(64)
        return h/max(h.sum(),1)
    raise ValueError(kind)

def run(cfg,data,folder):
    folder=Path(folder); folder.mkdir(exist_ok=True,parents=True); rows=[]
    for spec in cfg['classical']:
        kind,*classifier=spec.split(':'); clf_name=classifier[0] if classifier else 'svm'; bow=None; start=time.perf_counter()
        if 'siftbow' in kind:
            descriptors=[]
            # Bounded memory, all source descriptors strictly from train.
            rng=np.random.default_rng(cfg['seed'])
            for i in rng.permutation(len(data['train']))[:5000]:
                im,_=data['train'].raw(i); gray=cv2.cvtColor(image_array(im),cv2.COLOR_RGB2GRAY)
                _,desc=cv2.SIFT_create(nfeatures=64).detectAndCompute(gray,None)
                if desc is not None: descriptors.append(desc)
            if not descriptors or sum(len(x) for x in descriptors)<64: raise RuntimeError('Fewer than 64 train SIFT descriptors')
            bow=MiniBatchKMeans(64,random_state=cfg['seed'],n_init=3).fit(np.concatenate(descriptors))
        xs={}; ys={}
        for split in ['train','val']:
            ims=[data[split].raw(i) for i in range(len(data[split]))]
            xs[split]=np.asarray([feature(im,kind,bow) for im,y in ims]); ys[split]=[y for im,y in ims]
        extraction=time.perf_counter()-start
        for c in ([.1,1,10] if cfg['variant']==34 else [1]):
            estimator=KNeighborsClassifier(5) if clf_name=='knn' else LinearSVC(C=c,class_weight='balanced' if cfg['variant'] in [10,28,35] else None,max_iter=10000)
            pipeline=make_pipeline(StandardScaler(),estimator); start=time.perf_counter(); pipeline.fit(xs['train'],ys['train']); train_seconds=time.perf_counter()-start
            pred=pipeline.predict(xs['val']); encoded=pickle.dumps(dict(pipeline=pipeline,bow=bow))
            name=spec.replace(':','_').replace('+','_')+f'_C{c}'
            (folder/f'{name}.pkl').write_bytes(encoded)
            row=dict(name=name,accuracy=accuracy_score(ys['val'],pred),macro_F1=f1_score(ys['val'],pred,average='macro',zero_division=0),
                     report=classification_report(ys['val'],pred,output_dict=True,zero_division=0),feature_dim=xs['train'].shape[1],
                     train_seconds=train_seconds,feature_seconds=extraction,model_bytes=len(encoded))
            # Include feature extraction in end-to-end classic latency.
            samples=[data['val'].raw(i)[0] for i in range(min(100,len(data['val'])))]
            for im in samples[:20]: pipeline.predict([feature(im,kind,bow)])
            start=time.perf_counter()
            for im in samples: pipeline.predict([feature(im,kind,bow)])
            row['latency_ms']=(time.perf_counter()-start)*1000/len(samples)
            rows.append(row); dump(folder/'metrics.json',rows)
    return rows
