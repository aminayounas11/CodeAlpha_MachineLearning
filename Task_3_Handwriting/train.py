from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

ROOT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
SEED=42; torch.manual_seed(SEED);np.random.seed(SEED)
# %%
digits=load_digits() # 8x8 grayscale digits 0-9, packaged with scikit-learn
X=(digits.images.astype('float32')/16.0)[:,None,:,:]; y=digits.target.astype('int64')
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=.2,random_state=SEED)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,stratify=y_train,test_size=.15,random_state=SEED)
tr=DataLoader(TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train)),batch_size=64,shuffle=True)
val=DataLoader(TensorDataset(torch.from_numpy(X_val),torch.from_numpy(y_val)),batch_size=128)
test=DataLoader(TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test)),batch_size=128)
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*2*2,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x): return self.net(x)
model=DigitCNN();optim=torch.optim.Adam(model.parameters(),lr=.002);loss_fn=nn.CrossEntropyLoss()
def evaluation(loader):
    model.eval();yt=[];yp=[]
    with torch.no_grad():
        for xb,yb in loader:
            pred=model(xb).argmax(1); yt.extend(yb.numpy());yp.extend(pred.numpy())
    return np.asarray(yt),np.asarray(yp)
# %%
best=-1
for epoch in range(15):
    model.train()
    for xb,yb in tr:
        optim.zero_grad();loss=loss_fn(model(xb),yb);loss.backward();optim.step()
    yt,yp=evaluation(val);acc=accuracy_score(yt,yp)
    print(f'Epoch {epoch+1}: validation accuracy={acc:.4f}')
    if acc>best:
        best=acc;torch.save(model.state_dict(),OUT/'digit_cnn.pt')
model.load_state_dict(torch.load(OUT/'digit_cnn.pt',map_location='cpu',weights_only=True))
yt,yp=evaluation(test);acc=accuracy_score(yt,yp)
print('Held-out test accuracy:',round(acc,4));print(classification_report(yt,yp,zero_division=0))
pd.DataFrame([{'test_accuracy':acc,'validation_accuracy_best':best}]).to_csv(OUT/'metrics.csv',index=False)
ConfusionMatrixDisplay.from_predictions(yt,yp);plt.title('Digit CNN - held-out test');plt.savefig(OUT/'confusion_matrix.png',dpi=140,bbox_inches='tight');plt.close()
