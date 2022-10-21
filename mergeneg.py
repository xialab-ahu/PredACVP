from Imblearn import  read,split
from Base_Classifier import Select_feature
from Sort import sort_sfs
from Test import test_best,testmrmr
import  pandas as pd
import warnings
from Evaluation import write_train,write_test
from Train import L_Model
import numpy as np
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
if __name__ == "__main__":

    training_sets_pos = pd.read_csv("data/acvp-train.csv")
    training_sets = pd.read_csv("data/nonavp-train.csv")
    test_sets_pos = pd.read_csv("data/acvp-test.csv")
    test_sets = pd.read_csv("data/nonavp-test.csv")
    trainlen = training_sets_pos.shape[0] + training_sets.shape[0]
    testlen =  test_sets_pos.shape[0] + test_sets.shape[0]
    dataName = [ "non-AVP"]
    nlab1 = 'Gen'
    print('---------------------Use MRMR for forward search-------------------')
    scaler = StandardScaler()
    col = np.array([i for i in range(527)])
    for nlab in dataName:
        print("Make Classification for Anti-CoV versus {:s}".format(nlab))
        total_train = pd.concat([training_sets_pos,training_sets],axis=0)
        X_train = total_train.iloc[:,2:].values

        y_train = np.array([1 if i < training_sets_pos.shape[0] else 0 for i in range(trainlen)])

        total_test = pd.concat([test_sets_pos, test_sets], axis=0)
        X_test = total_test.iloc[:,2:].values

        print(X_test.shape)
        y_test = np.array([1 if i < test_sets_pos.shape[0] else 0 for i in range(testlen)])
        data,label = read(X_train, y_train,flag=True,name='anti')

        gen = pd.read_csv('./data/mergenmbalance/leak-nonavp.csv').iloc[:,1:]
        gen.columns =col
        print(gen)






        relpos = data.iloc[211:,:]
        relneg = data.iloc[:211,:]
        genneg = gen.iloc[:208,:]


        data1 = pd.concat([relpos,genneg,relneg],axis=0)
        print(data1.shape)
        lable1 = np.array([1 if i < 211 else  0 for i in range(data1.shape[0])])
        Train_feature = split(data1,lable1)
        print(Train_feature)
        Feature_select ,model_name = Select_feature(Train_feature, name=nlab1)
        Train_sfs, fea_max = sort_sfs( Feature_select)
        # Train_sfs.to_csv('./Anti/Train/Gen/leakbest.csv')
        train_best = L_Model(Train_sfs)
        write_train(nlab,train_best)

        Testdata,label = read(X_test,y_test,flag=False,name='test')
        Test_Feature = split(Testdata,label)
        TestXGB = testmrmr(Test_Feature,fea_max, name=nlab)
        print("Test Anti-CoV versus {:s}".format(nlab), " based on LR  results")
        test_result = test_best(TestXGB)
        write_test(nlab,test_result)
