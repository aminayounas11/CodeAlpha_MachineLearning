from pathlib import Path
import re
import random
import numpy as np
import pandas as pd
import librosa
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA=ROOT/'data'/'raw';OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
np.random.seed(42);torch.manual_seed(42);random.seed(42)
# RAVDESS emotion codes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
EMOTIONS={'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
# %%
files=sorted(DATA.rglob('*.wav')) if DATA.exists() else []
records=[]
for wav in files:
    parts=wav.stem.split('-')
    if len(parts)==7 and parts[0]=='03' and parts[2] in EMOTIONS:
        records.append((wav,EMOTIONS[parts[2]],parts[6]))
if not records:
    raise FileNotFoundError('No RAVDESS speech WAV files found in data/raw. Read Task_2_Speech_Emotion/README.md to download and extract RAVDESS Audio_Speech_Actors_01-24. This is required to train/evaluate real emotion labels.')
# Actor-disjoint train/test split avoids the same speaker appearing in both sets
paths,labels,actors=map(np.array,zip(*records));classes=sorted(set(labels));y=np.asarray([classes.index(x) for x in labels])
assert len(set(actors))>=4,'Need recordings from at least 4 actors for a meaningful speaker-independent split'
gss=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
train_idx,test_idx=next(gss.split(paths,y,groups=actors))
print('Found WAV files:',len(paths),'classes:',classes,'actors:',len(set(actors)))
# %%
def mfcc_features(path, sr=16000, seconds=3, n_mfcc=40):
    audio,_=librosa.load(str(path),sr=sr,mono=True,duration=seconds)
    target=sr*seconds
    audio=np.pad(audio[:target],(0,max(0,target-len(audio))))
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=n_mfcc,n_fft=512,hop_length=256)
    # Fix frame dimension so CNN sees equal-length inputs
    frames=188
    if mfcc.shape[1]<frames: mfcc=np.pad(mfcc,((0,0),(0,frames-mfcc.shape[1])))
    return mfcc[:,:frames].astype('float32')
features=np.stack([mfcc_features(p) for p in paths])
# Normalize using train-only moments (prevent test-set information leakage)
mean=features[train_idx].mean(axis=(0,2),keepdims=True)
std=features[train_idx].std(axis=(0,2),keepdims=True)+1e-6
features=(features-mean)/std
# Input shape [N, 40 MFCC features, 188 time frames]
tr=DataLoader(TensorDataset(torch.from_numpy(features[train_idx]),torch.from_numpy(y[train_idx]).long()),batch_size=32,shuffle=True)
te=DataLoader(TensorDataset(torch.from_numpy(features[test_idx]),torch.from_numpy(y[test_idx]).long()),batch_size=64)
# %%
class EmotionCNN(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(40,64,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(64,96,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(96,48),nn.ReLU(),nn.Dropout(.2),nn.Linear(48,n_classes))
    def forward(self,x):return self.net(x)
model=EmotionCNN(len(classes));optimizer=torch.optim.Adam(model.parameters(),lr=.001);criterion=nn.CrossEntropyLoss()
for epoch in range(18):
    model.train();losses=[]
    for xb,yb in tr:
        optimizer.zero_grad();loss=criterion(model(xb),yb);loss.backward();optimizer.step();losses.append(loss.item())
    print(f'Epoch {epoch+1}: training loss {np.mean(losses):.3f}')
# %%
model.eval();yt=[];yp=[]
with torch.no_grad():
    for xb,yb in te:
        pred=model(xb).argmax(1);yt.extend(yb.numpy());yp.extend(pred.numpy())
accuracy=accuracy_score(yt,yp);print('Actor-independent test accuracy:',accuracy)
print(classification_report(yt,yp,labels=list(range(len(classes))),target_names=classes,zero_division=0))
pd.DataFrame([dict(dataset='RAVDESS speech',test_accuracy=accuracy,n_test=len(yt),n_train=len(train_idx),split='actor_disjoint')]).to_csv(OUT/'metrics.csv',index=False)
torch.save({'state_dict':model.state_dict(),'classes':classes,'mfcc_mean':mean,'mfcc_std':std},OUT/'speech_emotion_cnn.pt')
ConfusionMatrixDisplay.from_predictions(yt,yp,labels=list(range(len(classes))),display_labels=classes,xticks_rotation=45);plt.savefig(OUT/'confusion_matrix.png',dpi=140,bbox_inches='tight');plt.close()
