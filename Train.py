import joblib
Randon_seed = 10
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
from sklearn.model_selection import  StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Randon_seed)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from Evaluation import scores,score_threshold
from pathlib import Path
import pandas as pd
Path('./Models1/Trainbest/').mkdir(exist_ok = True,parents = True)
model_name = "LR"

def L_Model(file1):
     # file1 = pd.read_csv(file1)
     mertics = []
     trainX = file1.iloc[:,1:].values
     trainY = file1.iloc[:,0].values
     for train_index,test_index in kf.split(trainX,trainY):
        kf_X_train,kf_y_train = trainX[train_index],trainY[train_index]
        kf_X_test,kf_y_test = trainX[test_index],trainY[test_index]
        clf = LR(random_state=Randon_seed)
        clf = clf.fit(kf_X_train,kf_y_train)
        y_pred = clf.predict_proba(kf_X_test)[:,1]
        scorel = scores(kf_y_test,y_pred)
        mertics.append(scorel)
     joblib.dump(clf,f'./Models1/Trainbest/{model_name}.model')
     return mertics



