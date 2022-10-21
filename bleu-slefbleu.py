

import matplotlib.pyplot as plt
import numpy as np
index = np.arange(5)
from matplotlib import rcParams

relacvp = np.array([ 0.978,0.857,0.624,0.463,0.731])
leakacvp = np.array([0.976,0.795,0.441,0.236,0.612])



plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文汉字
plt.rcParams['font.family'] = ['sans-serif']
bar_width = 0.2



relacvpSelfbleu = np.array([ 0.980,0.866,0.680,0.524,0.763])
leakacvpSelfbleu = np.array([0.977,0.818,0.508,0.299,0.651])
acvpSelfbleu = np.array([0.972,0.783,0.586,0.484,0.706])


import matplotlib.pyplot as plt
from matplotlib import gridspec



f = plt.figure(figsize=(20,6))
spec = gridspec.GridSpec(ncols=5, nrows=1)
ax0 = f.add_subplot(spec[0:2])
plt.title('A',fontproperties= 'Times New Roman',size=26,loc='left')
plt.bar(index-bar_width, leakacvp, width=0.2, color='r',label='LeakACVP')
plt.bar(index  ,relacvp,width=0.2,color='c',label='RelACVP')


plt.ylabel('Score',fontproperties= 'Times New Roman', size= 26)
y_score = np.arange(0.2,1,0.1)
plt.yticks(y_score,fontproperties= 'Times New Roman',size=26)
plt.ylim(0.2,1)
plt.xticks([0,1,2,3,4],['BLEU-2','BLEU-3','BLEU-4','BLEU-5','AVE'],fontproperties= 'Times New Roman',size=24)
plt.legend(prop ={'family': 'Times New Roman','size': 22},loc='upper right')

ax1 = f.add_subplot(spec[2:])
plt.title('B',fontproperties= 'Times New Roman',size=26,loc='left')
plt.bar(index- bar_width, leakacvpSelfbleu, width=0.2, color='r',label='LeakACVP')
plt.bar(index  ,relacvpSelfbleu,width=0.2,color='c',label='RelACVP')
plt.bar(index + bar_width ,acvpSelfbleu,width =0.2,color = 'b',label ='ACVP')

# plt.xlabel('Self-BLEU',fontdict={'family': 'Times New Roman', 'size': 16})
plt.ylabel('Score',fontproperties= 'Times New Roman', size= 26)
plt.yticks(y_score,fontproperties= 'Times New Roman',size=26)
plt.ylim(0.2,1)
plt.xticks([0,1,2,3,4],['self-BLEU-2','self-BLEU-3','self-BLEU-4','self-BLEU-5','AVE'],fontproperties= 'Times New Roman',size=24)
plt.legend(prop ={'family': 'Times New Roman','size': 22}, loc='upper right')

plt.show()