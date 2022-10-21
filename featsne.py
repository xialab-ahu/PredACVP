
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

from sklearn import metrics
import numpy as np

x = range(-60,60,10)
y = range(-80,80,10)

def result(file,label):
    res3 = []
    result = metrics.silhouette_score(file,labels=label,metric='euclidean')
    res3.append(result)
    return res3

f = plt.figure(figsize=(16,6))

leakacvp_nonavp_best = pd.read_csv('./Anti/Train/Gen/relbest.csv').iloc[:,2:]
x_feature_vector_acvpleakbest = TSNE().fit_transform(leakacvp_nonavp_best)
samples_count6 = leakacvp_nonavp_best.shape[0]
print(samples_count6)
anti_non6 = ['null']*samples_count6
for i in range(samples_count6):
    if i < 211:
        anti_non6[i] = 'ACVP'
    elif i < 414:
        anti_non6[i] = 'RelACVP'
    else:
        anti_non6[i] = 'non-ACVP'
acvp_best = leakacvp_nonavp_best.iloc[:211,:]
leakacvp_best = leakacvp_nonavp_best.iloc[211:419,:]
nonacvp_acvp_best = leakacvp_nonavp_best.iloc[419:630,:]
nonavp_leak_best = leakacvp_nonavp_best.iloc[630:,:]
x_feature_vector_acvpnpnavpleakbest = pd.DataFrame({'Dim1':x_feature_vector_acvpleakbest[:,0],'Dim2':x_feature_vector_acvpleakbest[:,1],'Category':anti_non6})

acvp_nonavp_best = pd.concat([acvp_best,nonacvp_acvp_best],axis=0)
leak_acvp_best = pd.concat([acvp_best,leakacvp_best],axis=0)
leak_nonavp_best  = pd.concat([leakacvp_best,nonavp_leak_best],axis=0)

acvp_nonavp_best_label = np.array([1 if i < 211 else 0 for i in range(422)])
leakacvp_nonavp_best_label = np.array([1 if i < 208 else 0 for i in range(416)])
leak_acvp_best_label = [1 if i < 211 else 0 for i in range(419)]

x_feature_vector_acvpnonavpbest = TSNE().fit_transform(acvp_nonavp_best)
x_feature_vector_leaknonavpbest = TSNE().fit_transform(leak_nonavp_best)
x_feature_vector_leakacvpbest = TSNE().fit_transform(leak_acvp_best)

acvp_nonavp_best_res = result(x_feature_vector_acvpnonavpbest,acvp_nonavp_best_label)
leakacvp_nonavp_best_res = result(x_feature_vector_leaknonavpbest,leakacvp_nonavp_best_label)
leak_acvp_best_res = result(x_feature_vector_leakacvpbest,leak_acvp_best_label)
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Category",
            data=x_feature_vector_acvpnpnavpleakbest,
)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xticks(x,fontproperties = 'Times New Roman', size = 24)
plt.yticks(y,fontproperties = 'Times New Roman', size = 24)
# plt.xlabel('Dim1',fontdict={'family' : 'Times New Roman', 'size'   : 20})
plt.xlabel('Dim1',fontdict={'family' : 'Times New Roman', 'size'   : 24})
plt.ylabel('Dim2',fontdict={'family' : 'Times New Roman', 'size'   : 24})
plt.rcParams.update({'font.size':10})
plt.text(-55,60,'ACVP VS non-ACVP'+ "=" + str(round(acvp_nonavp_best_res[0],3)),fontdict={'family': 'Times New Roman', 'size': 24})
plt.text(-55,50,'RelACVP VS nonACVP'+ "=" + str(round(leakacvp_nonavp_best_res[0],3)),fontdict={'family': 'Times New Roman', 'size': 24})
plt.text(-55,40,'ACVP VS RelACVP'+ "=" + str(round(leak_acvp_best_res[0],3)),fontdict={'family': 'Times New Roman', 'size': 24})
plt.legend(prop ={'family': 'Times New Roman','size': 20},loc='upper right')

plt.show()

