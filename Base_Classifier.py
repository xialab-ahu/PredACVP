from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd
import joblib
import sys
from numpy import  *
sys.path.append('../')
from  pathlib import  Path
from sklearn.model_selection import  StratifiedKFold
from xgboost.sklearn import XGBClassifier as XGBoost
from sklearn.linear_model import LogisticRegression as LR
Randon_seed = 10
njobs = 8

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
import numpy as np
from sklearn.neural_network import MLPClassifier as ANN
from sklearn.ensemble import ExtraTreesClassifier as ERT


Path(f'./Models1/').mkdir(exist_ok=True, parents=True)

def base_clf(clf,X_train,y_train,model_name,fea_name,n_folds=5):
    ntrain = X_train.shape[0]
    nclass = len(np.unique(y_train))
    base_train = np.zeros((ntrain, nclass))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Randon_seed)
    for train_index, test_index in kf.split(X_train,y_train):
        kf_X_train,kf_y_train = X_train[train_index],y_train[train_index]
        kf_X_test ,kf_y_test = X_train[test_index],y_train[test_index]
        clf.fit(kf_X_train, kf_y_train)
        base_train[test_index] = clf.predict_proba(kf_X_test)
    clf.fit(X_train,y_train)
    joblib.dump(clf, f'./Models1/{model_name}_{fea_name}.model')
    return base_train[:,-1]

ANN_clf = ANN(random_state=Randon_seed)
RF_clf = RF(random_state=Randon_seed)
KNN_clf = KNN(n_jobs=njobs)
ERT_clf = ERT(random_state=Randon_seed,n_jobs=njobs)
XGBoost_clf = XGBoost(random_state=Randon_seed)
LR_clf = LR(random_state=Randon_seed)
model_clf = [ANN_clf,RF_clf,KNN_clf,ERT_clf,XGBoost_clf]
model_name = ['ANN','RF','KNN','ERT','XGB']
def Select_feature(file,name):
    Path(f'./Anti/Train/{name}/').mkdir(exist_ok=True, parents=True)
    feature_name = ['AAC','DPC','CKSAAGP','PAAC','PHYC']
    result0 = []
    result = []
    for j in range(1,len(file)):
        label = file[0]
        data = file[j]
        for i in range(len(model_clf)):
            result.append(base_clf(model_clf[i],data,np.array(label),model_name[i],feature_name[j-1]))
            feaname = model_name[i] +'_' + feature_name[j-1]
            result0.append(feaname)
    Features = pd.DataFrame(np.array(result).T, columns=result0)
    y = pd.DataFrame(label,columns=['class'])
    TRFeatures = pd.concat([y,Features],axis=1,join='inner')
    file_path = (f'.\Anti\Train\{name}\Features.csv')
    TRFeatures.to_csv(file_path,index=False)
    return [file_path,result0]







