import xgboost as xgb
import time
import random
import numpy as np
from voting_rule import labeling
from scipy.special import entr
import matplotlib.pyplot as plt
import graphviz
from itertools import permutations
import copy
class active_learning:
    def get_group(candidates,preference_profile):
        group = active_learning.generate_by_axiom(candidates,preference_profile,generate_by_axioms.generate_by_neutrality,10)
        return group
    def choose_pf(candidates,preference_profiles,model_name, no_group =None):
        clf = xgb.Booster({'nthread': 4})  # init model
        clf.load_model(model_name)  # load data
        max_uncertainty = 0
        min_uncertainty = 100
        max_chosen = []
        min_chosen = []
        for pf in preference_profiles:
            #print(pf)
            if no_group == 1:
                pf_group = [pf]
            else:
                pf_group = active_learning.get_group(candidates,pf)
            #print(labeling.get_Xs(candidates,pf_group))
            uncertainty = active_learning.compute_uncertainty(clf,np.array(labeling.get_Xs(candidates,pf_group)))
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                max_chosen = pf
            if uncertainty < min_uncertainty:
                min_uncertainty = uncertainty
                min_chosen = pf 
        print(max_uncertainty)

        return max_chosen
    def generate_by_axiom(candidates,pf,axiom_generation_func,n,winner = None):
        if winner == None:
            pfs= axiom_generation_func(candidates,pf,n)
            return pfs
        else:
            pfs,winners= axiom_generation_func(candidates,pf,n,winner)
            return pfs,winners
    def initialize_model(new_Xs,new_ys,params,model_name):
        data = xgb.DMatrix(new_Xs, label=new_ys)
        clf = xgb.train(params,data)
        #clf = xgb.train(params,data,xgb_model=model_name)
        clf.save_model(model_name)
    def update_model(new_Xs,new_ys,params,model_name):
        data = xgb.DMatrix(new_Xs, label=new_ys)
        clf = xgb.train(params,data,xgb_model=model_name)
        clf.save_model(model_name)
        
    def compute_uncertainty(clf,Xs):
        distributions = clf.predict(xgb.DMatrix(Xs))
        #print(distributions)
        uncertainty = active_learning.entropy(distributions)
        return uncertainty
    def prediction_accuracy(Xs,true_ys,model_name):
        clf = xgb.Booster({'nthread': 4})  # init model
        clf.load_model(model_name)  # load data
        """
        xgb.plot_tree(clf,num_trees=0)
        xgb.plot_tree(clf,num_trees=1)
        xgb.plot_tree(clf,num_trees=2)
        plt.rcParams['figure.figsize'] = [50, 10]
        """
        #print(plt.show())
        #plt.ion()
        preds = clf.predict(xgb.DMatrix(Xs))
        preds = np.array(preds)
        preds = np.argmax(preds,axis = 1)
        error = np.mean( preds != true_ys )
        print("percentage Error:",error)
        return 1-error

    #get the average entropy for distributions
    def entropy(distributions):
        #print(entr(distributions).sum(axis=1).sum(axis=0)/len(distributions))
        return entr(distributions).sum()/len(distributions)

class generate_by_axioms:
    #sample n preference profiles to estimate the expected value of the entropy
    def generate_by_anonymity(candidates,pf,n,winner = None):
        pfs = []
        for i in range(n):
            new_pf = random.sample(pf, len(pf))
            pfs.append(new_pf)
        if winner == None:
            return pfs
        else:
            winners = [winner]*len(pfs)
            return pfs,winners
    def generate_by_neutrality(candidates,pf,n,winner = None):
        perms = list(permutations(candidates))
        perms = [list(ele) for ele in perms] 
        if winner!=None:
            winner_index = candidates.index(winner)
        #print(perms)
        pfs = []
        winners = []
        for perm in perms:
            new_pf = copy.deepcopy(pf)
            for j in range(len(new_pf)):
                ranking =  new_pf[j]
                new_ranking = copy.deepcopy(ranking)
                for c in candidates:
                    c_index = candidates.index(c)
                    new_c = perm[c_index]
                    index = ranking.index(c)
                    new_ranking[index] = new_c
                new_pf[j] = new_ranking
            if winner!=None:
                new_winner = perm[winner_index]
                winners.append(new_winner)
            pfs.append(new_pf)
        if winner == None:
            return pfs
        else:
            return pfs,winners