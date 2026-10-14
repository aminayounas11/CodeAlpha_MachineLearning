from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,ConfusionMatrixDisplay,classification_report

ROOT=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
# %%
data=load_breast_cancer(as_frame=True)
# Original sklearn labels: 0=malignant, 1=benign. Recode 1=malignant positive class.
X=data.data; y=(data.target==0).astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,random_state=42)),
        'Random Forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1)}
rows=[]
for name,model in models.items():
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_f1=cross_val_score(model,X_train,y_train,cv=cv,scoring='f1').mean()
    model.fit(X_train,y_train); p=model.predict(X_test);prob=model.predict_proba(X_test)[:,1]
    rows.append(dict(model=name,cv_f1=cv_f1,accuracy=accuracy_score(y_test,p),precision=precision_score(y_test,p),recall=recall_score(y_test,p),f1=f1_score(y_test,p),roc_auc=roc_auc_score(y_test,prob)))
    print(name,'\n',classification_report(y_test,p,target_names=['benign','malignant']))
    if name=='Logistic Regression':
        joblib.dump(model,OUT/'breast_cancer_model.joblib')
        ConfusionMatrixDisplay.from_predictions(y_test,p,display_labels=['benign','malignant']);plt.title('Holdout predictions');plt.savefig(OUT/'confusion_matrix.png',dpi=140,bbox_inches='tight');plt.close()
results=pd.DataFrame(rows);print(results.to_string(index=False));results.to_csv(OUT/'metrics.csv',index=False)
print('EDUCATIONAL EXAMPLE ONLY: not a medical device or diagnostic recommendation')
