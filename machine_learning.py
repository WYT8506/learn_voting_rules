import math
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
import sklearn.svm as svm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import BallTree
from sklearn.neighbors import DistanceMetric
import pandas as pd
import numpy as np
from scipy import spatial
import graphviz
from data import data_generator
from pulp import *
from itertools import permutations
import contextlib
import io
import sys
import time
import random
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
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=123)
        #data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
        #print(data_dmatrix)
        xg_reg = xgb.XGBClassifier(objective ='multi:softmax', learning_rate = 0.2,
                max_depth = 10, n_estimators = 30)#was 2 10 300
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
        print(plt.show())
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

        train_preds = clf.predict(X_sample)
        test_preds = clf.predict(X_test)

        count = 0
        for i in range(len(preds)):
            if true_y[i][0] != preds[i]:
                count+=1
        
        print("testing accuracy is:"+ str(1-count/len(preds)))

class SVM:
    def test_SVM(candidates, preference_profiles,preference_profiles_test,voting_rule):
        X_sample = SVM.get_histograms(candidates, preference_profiles)
        X_test = SVM.get_histograms(candidates, preference_profiles_test)
        Y_sample = SVM.get_labels(candidates,preference_profiles,voting_rule) 
        Y_test = SVM.get_labels(candidates,preference_profiles_test,voting_rule)
        SVM.train(X_sample,Y_sample,X_test,Y_test)

    def get_histograms(candidates,preference_profiles):
        histograms = []
        for pp in preference_profiles:
            hist = data_generator.histogram(candidates,pp)
            histograms.append([e[1] for e in hist])
        return histograms
    def get_labels(candidates,preference_profiles,voting_rule):
        labels = []
        for pp in preference_profiles:
             labels.append(voting_rule(candidates,pp))
        return labels
    def train(X_sample, Y_sample, X_test,Y_test):
#[[0], [1], [2], [3]]
#[0, 1, 2, 3]
        clf = svm.SVC(decision_function_shape='ovo',kernel='rbf',gamma = 0.001,C = 0.2)
        clf.fit(X_sample, Y_sample)
     
        sample_preds = clf.predict(X_sample)
        test_preds= clf.predict(X_test)
        count = 0
        for i in range(len(sample_preds)):
            if Y_sample[i] == sample_preds[i]:
                count+=1
        print(count/len(X_sample))

        count = 0
        for i in range(len(test_preds)):
            if Y_test[i] == test_preds[i]:
                count+=1
        print(count/len(X_test))

class KNN:

    def get_distance(candidates,preference_profile1,preference_profile2):
        sum_square = 0
        for i in range(len(preference_profile1)):
            rank_difference = 0
            for j in range(len(preference_profile1[0])):
                if preference_profile1[i][j]!=preference_profile2[i][j]:
                    rank_difference+=1
            sum_square += rank_difference
        return sum_square
    def ranking_distance(ranking1,ranking2):
        distance = 0
        for i in range(len(ranking1)):
            for j in range(i+1,len(ranking1)):
                c1 = ranking1[i]
                c2 = ranking1[j]
                if (ranking2.index(c1) > ranking2.index(c2)):
                    distance = distance + 1
        return distance
    def earth_mover_distance(h1,h2):
        candidates = []
        for x in range(10):
            if math.factorial(x) == len(h1):
                for i in range(x):
                    candidates.append(i)
                break
        perm = list(permutations(candidates))
        suppliers = []
        demanders = []
        supply = {}
        demand = {}
        for i in range(len(h1)):
            if (h1[i]> h2[i]):
                suppliers.append(i)
                supply[i] = h1[i] - h2[i]
            elif h1[i]<h2[i]:
                demanders.append(i)
                demand[i] = h2[i] - h1[i]

        if(len(suppliers)==0 and len(demanders)==0):return 0 

        costs = {}
        for supplier in suppliers:
            d = {}
            for demander in demanders:
                d[demander]=KNN.ranking_distance(perm[supplier],perm[demander])
            costs[supplier] = d
        prob = LpProblem("Distribution Problem",LpMinimize) 
        Routes = [(w,b) for w in suppliers for b in demanders]
        
        route_vars = LpVariable.dicts("Route",(suppliers,demanders),0,None,LpInteger)
        
        prob += lpSum([route_vars[w][b]*costs[w][b] for (w,b) in Routes])
        
        # The supply maximum constraints are added to prob for each supply node 
        for w in suppliers:
            prob += lpSum([route_vars[w][b] for b in demanders]) <= supply[w], "Sum of Products out of Warehouse %s"%w

        # The demand minimum constraints are added to prob for each demand node (bar)
        for b in demanders:
            prob += lpSum([route_vars[w][b] for w in suppliers]) >= demand[b], "Sum of Products into Bars %s"%b

        total_distance = 0
        
        #prob.writeLP("TransportationProblem.lp")  # optional
        prob.solve(PULP_CBC_CMD(msg=0))

        total_distance = 0
        for i in range(len(prob.variables())):
            v = prob.variables()[i]
            #print(v.name, "=", v.varValue)
            total_distance=total_distance + v.varValue*costs[Routes[i][0]][Routes[i][1]]
            #print(total_distance)
        """
        print(perm)
        print(supply)
        print(demand)
        print(costs)
        print(Routes)
        print(route_vars)

        """
        #print("fuck")
        return total_distance

    def total_variation_distance(h1,h2):
        normalized_h1 = h1
        normalized_h2 = h2
        sum_1 = sum(normalized_h1)

        total_distance = 0
        for i in range(len(h1)):
            total_distance+=abs(normalized_h1[i]-normalized_h2[i])
        total_distance = total_distance/sum_1
        #print(h1,h2,total_distance)
        return total_distance/2
 
    def get_winner(candidates,preference_profiles,labels,preference_profile,voting_rule): 
        print("1:",preference_profile)
        min_distance = KNN.get_distance(candidates,preference_profiles[0],preference_profile)
        min_indexes = []
        winners = []
        distance_sum = 0
        for i in range(len(preference_profiles)):
            distance = KNN.get_distance(candidates,preference_profiles[i],preference_profile)
            if distance < min_distance:
                min_distance=distance
                min_indexes = []
                min_indexes.append(i)

            elif distance == min_distance:
                min_indexes.append(i)

            distance_sum+=distance
        for index in min_indexes:
            winners.append(labels[index])

        winner = max(set(winners), key = winners.count)

        print(min_indexes)
        print(winners)
        print(min_distance)
        print(distance_sum/len(preference_profiles))
        print("2:",preference_profiles[min_indexes[0]])
        return winner
    def build_ball_tree(histograms,distance_type): 

        tree = BallTree(np.array(histograms), leaf_size=1,metric = distance_type)
        return tree

    def ball_tree_query(tree,query_histogram,K):

        dist, ind = tree.query(np.array([query_histogram]), k=K)
        print(ind[0])
        return list(ind[0])


    def ball_tree_test(candidates,preference_profiles,preference_profiles_test,voting_rule,distance_type,k):
        labels = []
        test_labels = []
        data_histograms = []
        test_histograms = []
        for pp in preference_profiles:
            labels.append(voting_rule(candidates,pp))
            hist = data_generator.histogram(candidates,pp)
            data_histograms.append([e[1] for e in hist])

        for pp in preference_profiles_test:
            hist = data_generator.histogram(candidates,pp)
            test_histograms.append([e[1] for e in hist])

        print(len(data_histograms))
        tree = KNN.build_ball_tree(data_histograms,distance_type)



        count =0
        n = 1000
        for i in range(n):
            indexs = KNN.ball_tree_query(tree,test_histograms[i],k)
            winners = [labels[index] for index in indexs]
            print(winners)
            winner = max(set(winners), key = indexs.count)
            print(winner)
            print("this histogram is:",test_histograms[i])
            #print("closest_histogram is:",data_histograms[index])
            if winner == voting_rule(candidates,preference_profiles_test[i]):
                count+=1
                print('correct')
            else:print('wrong')
            print(i/n,"%")
        print(count/n)
        return(count/n)
    
    def euclideanDistance(x,y):
            return math.sqrt(sum([(a-b)**2 for (a,b) in zip(x,y)]))
    def test_distance_calculation_time(candidates,preference_profiles,n,distance_type):
        histograms = []
        for pp in preference_profiles:
            hist = data_generator.histogram(candidates,pp)
            histograms.append([e[1] for e in hist])
        start_time = time.time()
        for i in range(n):
            h1 = random.choice(histograms)
            h2 = random.choice(histograms)
            distance_type(h1,h2)
        print("---  seconds ---", (time.time() - start_time)/n)
    





    