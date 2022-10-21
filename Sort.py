import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import subprocess
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as RF
seed = 1
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
from sklearn.linear_model import  LogisticRegression
 	# 创建一个输出文件夹
LR = LogisticRegression(random_state=seed)
RF_clf = RF(random_state=seed)
def resort_training_feature_with_mRMR(file):
    SFS_out = 'temp/sfs_out'
    Path(SFS_out).mkdir(exist_ok=True,parents=True)
    df = pd.read_csv(file)
    shape = df.shape[1]

    mrmr_path = 'mrmr.exe'
    output_file = f"{SFS_out}/{Path(file).stem}-mrmr_out.txt"
    mrmr_commd = f"{mrmr_path} -i {file} -n {shape-1} -s 2000 > {output_file}"
    subprocess.call(mrmr_commd, shell = True)


    with open(output_file) as f:
        while not f.readline().startswith('*** mRMR features ***'):
            continue
        mRMR_top = []
        for i in f:
            if len(i) <= 1:
                break
            else:
                elemt = i.split()
                mRMR_top.append(elemt)
    feature_df = pd.read_csv(file)
    mRMR_top_index = list(map(int,pd.DataFrame(mRMR_top).iloc[1:,1]))
    mRMR_top_index.insert(0,0)
    features_top_df = feature_df.iloc[:, mRMR_top_index]
    mRMR_top_index = mRMR_top_index[1:]
    return features_top_df.copy(),mRMR_top_index



def sfs(X_train,y_train,clf):
    auc_mean=[]
    for i in range(X_train.shape[1]):
        X_new=X_train[:,:i+1]
        scores=cross_val_score(clf,X_new,y_train,cv=cv,scoring="roc_auc",n_jobs=5)
        auc_mean.append(round(scores.mean(),2))
    return auc_mean

# Path(f"./Plot/true/").mkdir(exist_ok=True)
def plot_sfs(auc_mean,metrics,fea,plot_name='SFS'):
    fig=plt.figure(figsize=(20,12))
    ax = fig.add_subplot(1, 1, 1)
    x=range(1,len(auc_mean)+1)
    plt.plot(x, auc_mean, color='r', markerfacecolor='blue', marker='*')
    for a, b in zip(x, auc_mean):
        plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=14,color='red')
    plt.plot(x,auc_mean,'ro-', color='#4169E1', alpha=1, label=metrics,linewidth=4,ms=12)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0),)
    plt.grid(ls='--')
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper left')
    plt.title(plot_name,fontsize=18)
    plt.tick_params(axis='both',which='major',labelsize=12,)
    plt.ylabel('Performance',fontsize=16)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    ax.set_xticklabels([fea[i] for i in range(len(fea))], rotation=30, fontproperties = 'Times New Roman', size = 12)

    #保存文件

    # plt.savefig(f"./Plot/true/{metrics}-{plot_name}.png",format="png")
    plt.show()


def sort_sfs(file):
    Sortsfs ,mrmr_index = resort_training_feature_with_mRMR(file)
    fea_name = list(Sortsfs)
    fea_name = fea_name[1:]
    X1 = Sortsfs.iloc[:,1:].values
    y = Sortsfs.iloc[:,0].values
    clf = LR
    X2 = sfs(X1,y,clf)
    m='auc'
    plot_sfs(X2,m,fea_name,plot_name='relSFS')
    i = X2.index(max(X2))
    Sortsfsbest = Sortsfs.iloc[:, :i + 2]
    fea_max = fea_name[:i+1]
    return Sortsfsbest,fea_max
