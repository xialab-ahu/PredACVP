from imblearn.under_sampling import NearMiss,RandomUnderSampler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
def read(X,y,flag,name):
    if(flag):
        nm3 = NearMiss(version=3)
        # nm3 = RandomUnderSampler()
        data ,label = nm3.fit_sample(X,y)
        data = pd.DataFrame(data)
        print(label)
        data.to_csv(f'./data/mergenmbalance/{name}-nonavp.csv')
        temp = []
        for i in X.tolist():
            if i in np.array(data).tolist():
                continue
            else:
                temp.append(i)
        temp = pd.DataFrame(temp)
        temp.to_csv(f'./data/mergeremainnonavp/{name}-nonavpremain.csv')

    else:
        data = pd.DataFrame(X)
        label = y

    return data,label




def split(file,label):


    AAC = file.iloc[:, :20].values
    DPC = file.iloc[:, 20:420].values
    CKSAAGP = file.iloc[:, 420:495].values
    PAAC = file.iloc[:,495:519].values
    PHYC = file.iloc[:,519:527].values

    return [label,AAC,DPC,CKSAAGP,PAAC,PHYC]


