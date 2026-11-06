import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
d=load_breast_cancer()
# sklearn: 0=malignant,1=benign -> make malignant=1 positive class
X=d.data; y=1-d.target
print(X.shape, y.mean(), np.isnan(X).sum(), pd.DataFrame(X).duplicated().sum())
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
models={'LR':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
'DT':DecisionTreeClassifier(random_state=42),
'RF':RandomForestClassifier(n_estimators=200,random_state=42)}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
for n,m in models.items():
    m.fit(Xtr,ytr); p=m.predict(Xte)
    cvs=cross_val_score(m,X,y,cv=cv)
    print(n,round(accuracy_score(yte,p),4),round(precision_score(yte,p),4),round(recall_score(yte,p),4),round(f1_score(yte,p),4),confusion_matrix(yte,p).tolist(),round(cvs.mean(),4),round(cvs.std(),4))
rf=models['RF']
imp=sorted(zip(rf.feature_importances_,d.feature_names),reverse=True)[:5]
print([(round(a,3),b) for a,b in imp])