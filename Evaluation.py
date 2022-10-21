
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,fbeta_score,auc
from sklearn.metrics import  accuracy_score, precision_recall_curve


from sklearn.metrics import roc_curve,f1_score
import numpy as np
import os
import time

if not os.path.exists("results_classification"):
    os.mkdir("results_classification")
time_now = int(round(time.time() * 1000))
time_now = time.strftime("%Y-%m-%d_%H-%M", time.localtime(time_now / 1000))
cls_dir = "results_classification/Train/{}".format(time_now)
cls_dir_test = "results_classification/Test/{}".format(time_now)
os.makedirs(cls_dir)
os.makedirs(cls_dir_test)





def scores(y_test,y_pred,th=0.5):
    y_predlabel=[(0 if item< th else 1) for item in y_pred]
    tn,fp,fn,tp=confusion_matrix(y_test,y_predlabel).flatten()
    fpr, tpr, th = roc_curve(y_test, y_pred)
    SEN=tp*1./(tp+fn)
    SPE=tn*1./(tn+fp)
    F1 = f1_score(y_test, y_predlabel)
    MCC=matthews_corrcoef(y_test,y_predlabel)
    Acc=accuracy_score(y_test, y_predlabel)
    AUC=roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return [SEN,SPE,F1,MCC,Acc,AUC]


def score_threshold(y_test,y_pred):
    fpr, tpr, thresholds =roc_curve(y_test, y_pred)
    Gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(Gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(Gmean[index], ndigits=4)
    return thresholdOpt,gmeanOpt


def write_train(datasets,file):
    SEN = []
    SPE = []
    F1 = []
    MCC = []
    Acc = []
    AUC = []

    for i in range(len(file)):
        SEN.append(file[i][0])
        SPE.append(file[i][1])
        F1.append(file[i][2])
        MCC.append(file[i][3])
        Acc.append(file[i][4])
        AUC.append(file[i][5])


    with open(os.path.join(cls_dir, "result.txt"), 'a') as f:
        f.write('Datasets' + '    ' + 'SEN' + '     ' + 'SPE' + '    ' +'F1'+ '   '+ 'MCC' + '    ' +'Acc '  + '   ' + 'AUC' +'\n')
        f.write(str('Anti')+'-'+str(datasets)  + '  ' + str(format(np.mean(SEN),'.3f'))+ ' ' + str(format(np.mean(SPE),'.3f'))+ ' ' +str(format(np.mean(F1),'.3f'))+' '+ str(format(np.mean(MCC),'.3f')) +' '+str(format(np.mean(Acc),'.3f')) +' '+ str(format(np.mean(AUC),'.3f')) + '\n')
    for i in range(len(file)):
        with open(os.path.join(cls_dir, "result1.txt"), 'a') as f:
            f.write(
                'Datasets' + '    ' + 'SEN' + '     ' + 'SPE' + '    ' + 'F2' + '   ' + 'MCC' + '    ' + 'Acc ' + '   ' + 'AUC' + '\n')

            f.write(str('Anti') + '-' + str(datasets) + '  ' + str(format(SEN[i], '.3f')) + ' ' + str(
                format(SPE[i], '.3f')) + ' ' + str(format(F1[i], '.3f')) + ' ' + str(format(MCC[i], '.3f')) + ' ' + str(
                format(Acc[i], '.3f')) + ' ' + str(format(AUC[i], '.3f')) + '\n')



def write_test(datasets,file):
    with open(os.path.join(cls_dir_test,'result.txt'),'a') as f:
        f.write('Datasets' + '    ' + 'SEN' + '     ' + 'SPE' + '    ' +'F1'+ '  '+ 'MCC' + '   ' +'Acc' +'  '+ 'AUC'  +'\n')
        f.write(str('Anti')+'-'+str(datasets)  + '  ' + str(format(file[0], '.3f')) + ' ' + str(
                format(file[1], '.3f')) + ' ' + str(format(file[2], '.3f')) + ' ' + str(format(file[3], '.3f')) + ' ' + str(format(file[4], '.3f'))+' '+ str(format(file[5], '.3f'))  +'\n')