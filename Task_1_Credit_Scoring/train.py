from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUT = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)
SEED = 42
# %%
def make_demo_credit(n=3500, random_state=42):
    """SIMULATED illustrative patterns. This is NOT bank/credit bureau data."""
    rng=np.random.default_rng(random_state)
    age=rng.integers(18,75,n)
    income=np.clip(rng.normal(65000,27000,n),12000,250000)
    debt_to_income=rng.uniform(.03,.85,n)
    missed_payments=rng.poisson(1.0,n)
    credit_history_years=np.minimum(age-18,rng.uniform(0,20,n))
    employment_type=rng.choice(['salaried','self_employed','unemployed'],n,p=[.65,.25,.10])
    # probability: greater debts/missed payments increase simulated risk
    logit=-2 + 3.4*debt_to_income + .72*missed_payments - .000012*income - .07*credit_history_years + (employment_type=='unemployed')*.55
    probability=1/(1+np.exp(-logit))
    default=rng.binomial(1,probability)
    return pd.DataFrame(dict(age=age,income=income,debt_to_income=debt_to_income,missed_payments=missed_payments,credit_history_years=credit_history_years,employment_type=employment_type,default=default))

csv=os.environ.get('CREDIT_CSV','').strip()
df=pd.read_csv(csv) if csv else make_demo_credit()
if 'default' not in df.columns: raise ValueError('Dataset must contain binary target named default (0 or 1).')
if set(df['default'].dropna().unique()) - {0,1}: raise ValueError('default must be binary 0/1.')
X=df.drop(columns=['default']); y=df['default'].astype(int)
assert y.nunique()==2
print('Dataset:', 'USER-PROVIDED' if csv else 'SYNTHETIC DEMO', 'rows:',len(df))
# %%
num=X.select_dtypes(include=np.number).columns.tolist(); cat=X.columns.difference(num).tolist()
pre=ColumnTransformer([
 ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
 ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)
])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
results=[]
for name,est in [('Logistic Regression',LogisticRegression(max_iter=1000,random_state=SEED)),('Random Forest',RandomForestClassifier(n_estimators=150,min_samples_leaf=4,random_state=SEED,n_jobs=-1))]:
    pipe=Pipeline([('preprocessor',pre),('model',est)])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
    results.append(dict(model=name,accuracy=accuracy_score(y_test,pred),precision=precision_score(y_test,pred,zero_division=0),recall=recall_score(y_test,pred,zero_division=0),f1=f1_score(y_test,pred,zero_division=0),roc_auc=roc_auc_score(y_test,prob)))
    if name=='Random Forest':
        joblib.dump(pipe,OUT/'credit_model.joblib')
        ConfusionMatrixDisplay.from_predictions(y_test,pred);plt.title('Synthetic credit risk - RF');plt.savefig(OUT/'confusion_matrix.png',dpi=140,bbox_inches='tight');plt.close()
metrics=pd.DataFrame(results);print(metrics.to_string(index=False));metrics.to_csv(OUT/'metrics.csv',index=False)
