import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from Evaluation import scores



def test(file,model_name,name):
    Path(f'./Anti/Test/{name}/').mkdir(exist_ok=True, parents=True)
    y_label = file[0]
    feature_name = ['AAC','DPC','CKSAAGP','PAAC','PHYC']
    result1 = []
    feature_name_select = []
    for i in range(len(model_name)):
        feature_name_select.append(model_name[i])

    for j in range(len(feature_name)):
        feature_each = file[j+1]
        for k in range(len(feature_name_select)):
            print(feature_name_select[k])
            if(feature_name[j] == feature_name_select[k].split('_')[1]):
                print(f'./Models/{feature_name_select[k]}.model')
                model = joblib.load(f'./Models1/{feature_name_select[k]}.model')
                model_name[k] = model.predict_proba(feature_each)[:,1]
                result1.append(model_name[k])
    a = result1
    Features = pd.DataFrame(np.array(a).T,columns=feature_name_select)
    y_label = pd.DataFrame(y_label, columns=['class'])
    Test = pd.concat([y_label,Features],axis=1,join='inner')
    file_path = (f'.\Anti\Test\{name}\Features.csv')
    Test.to_csv(file_path, index=False)
    return Test



def testmrmr(file, model_name,name):
    Path(f'./Anti/Test/mrmr/{name}/').mkdir(exist_ok=True, parents=True)
    y_label = file[0]
    result1 = []
    # 使用的特征
    fea = ['AAC','DPC','CKSAAGP','PAAC','PHYC']
    # 最优的特征组合
    feature_name_select = []
    for i in range(len(model_name)):
        feature_name_select.append(model_name[i])


    # 测试集加载最优的特征
    for k in range(len(feature_name_select)):
        for j in range(len(fea)):
            if(feature_name_select[k].split('_')[1] == fea[j]):
                print(feature_name_select[k].split('_')[1])
                feature_each = file[j + 1]
                print(feature_each.shape)
                model = joblib.load(f'./Models1/{feature_name_select[k]}.model')
                model_name[k] = model.predict_proba(feature_each)[:,1]
                result1.append(model_name[k])
            else:
                continue

    a = result1
    Features = pd.DataFrame(np.array(a).T, columns=feature_name_select)
    y_label = pd.DataFrame(y_label, columns=['class'])
    Test = pd.concat([y_label, Features], axis=1, join='inner')
    file_path = (f'.\Anti\Test\mrmr\{name}\Features.csv')
    Test.to_csv(file_path, index=False)
    return Test
model_name = 'LR'
def test_best(file):
    X = file.iloc[:,1:]
    y = file.iloc[:,0]
    model = joblib.load(f'./Models1/Trainbest/{model_name}.model')
    test_pred = model.predict_proba(X)[:,1]
    score = scores(y,test_pred)
    return score



