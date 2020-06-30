import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pandas as pd
import graphviz
#os.environ["PATH"] += os.pathsep+'C:\Users\wangy45\anaconda3\Library\bin'


class learn:
    def load_x(headers,sample_set):
        x_data = pd.DataFrame(sample_set)
        return x_data
    def load_y(candidates,labels):
        y = []
        for label in labels:
           y.append(candidates.index(label)) 
        y_data = pd.DataFrame(y)
        return y_data
    def train(x_data,y_data):
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=123)
        #data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
        #print(data_dmatrix)
        xg_reg = xgb.XGBClassifier(objective ='multi:softmax', colsample_bytree = 0.3, learning_rate = 0.2,
                max_depth = 2, alpha = 10, n_estimators = 300)#was 2 10 300
        xg_reg.fit(X_train,np.ravel(y_train))
        xg_reg.save_model('0001.model')
        pickle.dump(xg_reg, open("pima.pickle.dat", "wb"))
        #bst = xgb.train(param, dtrain, num_round, evallist)
        preds = xg_reg.predict(X_test)
        true_y = y_test.values.tolist()
        
        xgb.plot_tree(xg_reg,num_trees=0)
        xgb.plot_tree(xg_reg,num_trees=1)
        xgb.plot_tree(xg_reg,num_trees=2)
        plt.rcParams['figure.figsize'] = [50, 10]
        #print(plt.show())
        #plt.ion()
        
        xgb.plot_importance(xg_reg)
        #print(plt.show())
        plt.rcParams['figure.figsize'] = [5, 5]

        train_y_list = y_train.values.tolist()
        train_preds = xg_reg.predict(X_train)

        count = 0
        for i in range(len(train_preds)):
            if train_y_list[i][0] != train_preds[i]:
                count+=1
        print("training accuracy is:"+ str(1-count/len(train_preds)))

        count = 0
        for i in range(len(preds)):
            if true_y[i][0] != preds[i]:
                count+=1
        
        print("testing accuracy is:"+ str(1-count/len(preds)))
    def one_vs_one(x_data,y_data):
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=123)
        clf = OneVsOneClassifier( RandomForestClassifier(max_depth=7, n_estimators=15, max_features=20)).fit(X_train, y_train.values.ravel())

        train_y_list = y_train.values.tolist()
        train_preds = clf.predict(X_train)

        count = 0
        for i in range(len(train_preds)):
            if train_y_list[i][0] != train_preds[i]:
                count+=1
        print("training accuracy is:"+ str(1-count/len(train_preds)))
        
        preds = clf.predict(X_test)
        true_y = y_test.values.tolist()
        count = 0
        for i in range(len(preds)):
            if true_y[i][0] != preds[i]:
                count+=1
        
        print("testing accuracy is:"+ str(1-count/len(preds)))
        





    