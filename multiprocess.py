
import numpy as np
import random
import math
from IPython.display import clear_output
from collections import deque
import progressbar

import gym
from gym import spaces
#from stable_baselines.common.env_checker import check_env

import tensorflow.keras as keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

from scipy.special import entr
import os

from voting_rule_new import voting_rules
from voting_rule_new import labeling
import xgboost as xgb
from data_new import data_generator
import random
import pickle
import contextlib
import io
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from active_new import active_learning
from active_new import generate_by_axioms
from itertools import permutations
import copy

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecNormalize
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
#from stable_baselines.common import make_vec_env
from stable_baselines import results_plotter
#from monitor_ import Monitor
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
#from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from pposgd_simple import PPO1
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
import tensorflow as tf
xgb.set_config(verbosity=0)

def get_X_histograms(Xs,n_voters):
    histograms = []
    lists = []
    for i in range(len(Xs[0])):
        l = copy.deepcopy([])
        for X in Xs:
            l.append(X[i])
        lists.append(l)
        
    bin_array = list(range(n_voters+1))
    for l in lists:
        histograms.append(np.histogram(l, bins=bin_array, density = True)[0])    
    return np.concatenate(histograms).ravel()

def get_X_histograms_long(Xs,ys,n_voters):
    histograms = []
    lists = []
    for i in range(len(Xs[0])):
        l1 = copy.deepcopy([])
        l2 = copy.deepcopy([])
        l3 = copy.deepcopy([])
        for k in range(len(Xs)):
            X = Xs[k]
            if ys[k] == 0:
                l1.append(X[i])
            if ys[k] == 1:
                l2.append(X[i])
            else:
                l3.append(X[i])
            
        lists.append(l1)
        lists.append(l2)
        lists.append(l3)
        
    bin_array = list(range(n_voters+1))
    for l in lists:
        histograms.append(np.histogram(l, bins=bin_array, density = True)[0])    
    #print(len(histograms))
    #print(histograms)
    return np.concatenate(histograms).ravel()

def get_y_histograms(ys,n_candidates):
    bin_array = list(range(n_candidates+1))
    return np.histogram(ys, bins=bin_array, density = True)[0]
def get_pool_mean(Xs,X_size):
    if (len(Xs)==0):
        return np.array([0]*X_size)
    return np.mean(Xs,axis = 0)

def Condorcet(candidates,preference_profile):
    weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
    #print(preference_profile)
    #print(weighted_majority_graph)
    for i in range(len(weighted_majority_graph)):
        not_win = 0
        for j in range(len(weighted_majority_graph[0])):
            if weighted_majority_graph[i][j] == 0:
                not_win +=1
        if not_win ==1:
            return i
    return None

def neutrality_satisfaction_model(candidates,preference_profiles,n,model_name):
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    total_count = 0
    true_count = 0
    while 1:
        pp = random.choice(preference_profiles)
        old_Xs=np.array(labeling.get_Xs(candidates,[pp]))
        pp_win =np.argmax(clf.predict(xgb.DMatrix(old_Xs)),axis = 1)[0]

        perms = list(permutations(candidates))
        perms = [list(ele) for ele in perms] 
        #print(perms)

        for perm in perms:
            new_pp = copy.deepcopy(pp)
            for j in range(len(new_pp)):
                ranking =  new_pp[j]
                new_ranking = copy.deepcopy(ranking)
                for c in candidates:
                    c_index = candidates.index(c)
                    new_c = perm[c_index]
                    index = ranking.index(c)
                    new_ranking[index] = new_c
                new_pp[j] = new_ranking

            new_Xs=np.array(labeling.get_Xs(candidates,[new_pp]))
            new_pp_win = np.argmax(clf.predict(xgb.DMatrix(new_Xs)),axis = 1)[0]

            if new_pp_win != perm[candidates.index(pp_win)]:
                total_count += 1
            else:
                true_count +=1
                total_count+=1

        if(total_count > n):
            break
    return true_count/total_count

def condorcet_satisfaction(Xs,true_ys,model_name):
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    preds = clf.predict(xgb.DMatrix(Xs))
    preds = np.array(preds)
    preds = np.argmax(preds,axis = 1)
    error = np.mean( preds != true_ys )
    #print("percentage Error:",error)
    return 1-error
    """
    pfs_condorcet = np.array(pfs_condorcet)
    winners_condorcet = np.array(winners_condorcet)
    if model == None:
        labels = labeling.get_labels(candidates,pfs_condorcet,voting_rule)
        error = np.mean( labels != np.array(winners_condorcet) )
        return 1-error
    else :
        clf = xgb.Booster({'nthread': 4})  # init model
        clf.load_model(model)  # load data
        Xs = labeling.get_Xs(candidates,pfs_condorcet)
        print(Xs[0:3])
        preds = clf.predict(xgb.DMatrix(Xs))
        print(preds[0:20])
        preds = np.array(preds)
        preds = np.argmax(preds,axis = 1)
        print(preds[0:20])
        print(winners_condorcet[0:20])
        error = np.mean( preds != np.array(winners_condorcet) )
        return 1-error
    """
def consistency_satisfaction_model_process(candidates,preference_profiles,n,clf):
    print("start model")
    sys.stdout.flush()
    time1 = time.time()
    pred_time = 0

    profile_count = 0
    consistency_count = 0
    new_pps = []
    while profile_count<n:
        pp1 = random.choice(preference_profiles)
        pp2 = random.choice(preference_profiles)
        new_pp = pp1+pp2
        pp1_Xs=np.array(labeling.get_Xs(candidates,[pp1]))
        pp2_Xs=np.array(labeling.get_Xs(candidates,[pp2]))
        new_pp_Xs=np.array(labeling.get_Xs(candidates,[new_pp]))
        t1 = time.time()
        preds_pp1=np.argmax(clf.inplace_predict(pp1_Xs),axis = 1)
        preds_pp2=np.argmax(clf.inplace_predict(pp2_Xs),axis = 1)
        preds_newpp=np.argmax(clf.inplace_predict(new_pp_Xs),axis = 1)
        t2 = time.time()
        pred_time+=t2-t1
        
        if preds_pp1==preds_pp2:
            profile_count+=1
            if preds_pp1==preds_newpp:
               consistency_count+=1
    time2 = time.time()
    print(time2-time1,pred_time)
    return(consistency_count/profile_count)

def consistency_satisfaction_model(candidates,preference_profiles,n,model_name):

    time1 = time.time()
    pred_time = 0
    clf = xgb.Booster({'nthread': 1})  # init model
    clf.load_model(model_name)
    profile_count = 0
    consistency_count = 0
    new_pps = []
    while profile_count<n:
        pp1 = random.choice(preference_profiles)
        pp2 = random.choice(preference_profiles)
        new_pp = pp1+pp2
        pp1_Xs=np.array(labeling.get_Xs(candidates,[pp1]))
        pp2_Xs=np.array(labeling.get_Xs(candidates,[pp2]))
        new_pp_Xs=np.array(labeling.get_Xs(candidates,[new_pp]))
        t1 = time.time()
        preds_pp1=np.argmax(clf.predict(xgb.DMatrix(pp1_Xs)),axis = 1)
        preds_pp2=np.argmax(clf.predict(xgb.DMatrix(pp2_Xs)),axis = 1)
        preds_newpp=np.argmax(clf.predict(xgb.DMatrix(new_pp_Xs)),axis = 1)
        t2 = time.time()
        pred_time+=t2-t1
        
        if preds_pp1==preds_pp2:
            profile_count+=1
            if preds_pp1==preds_newpp:
               consistency_count+=1
    time2 = time.time()
    print(time2-time1,pred_time)
    return(consistency_count/profile_count)

def consistency_satisfaction_model_gpu(candidates,preference_profiles,n,model_name):
    """
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  
    """
    clf = xgb.Booster({'nthread': 1})  # init model
    clf.load_model(model_name)
    """
    q = mp.Queue()
    p = mp.Process(target=consistency_satisfaction_model_process, args=(candidates,preference_profiles,n,clf,q))
    p.start()
    return_value = q.get()
    p.join()
    """
    from multiprocessing import Pool
    with Pool() as pool:
        L = pool.starmap(consistency_satisfaction_model_process, [(candidates,copy.deepcopy(preference_profiles),int(n/8),clf.copy())]*8)
        print(L)
        return (L[0]+L[1]+L[2]+L[3]+L[4]+L[5]+L[6]+L[7])/8

def consistency_satisfaction_model_quick(candidates,preference_profiles,n,model_name):
    clf = xgb.Booster({'nthread': 1})  # init model
    clf.load_model(model_name)
    groups_pfs = []
    groups_winners= []
    for c in candidates:
        groups_pfs.append([])
        groups_winners.append([])


    for i in range(1000):
        pp1 = random.choice(preference_profiles)
        pp1_Xs=np.array(labeling.get_Xs(candidates,[pp1]))
        preds_pp1=np.argmax(clf.predict(xgb.DMatrix(pp1_Xs)),axis = 1)
        groups_pfs[preds_pp1].append(pp1)
        groups_winners[preds_pp1].append(preds_pp1)

    from multiprocessing import Pool
    with Pool() as pool:
        parm_list = [
        (candidates,copy.deepcopy(groups_pfs[0]),copy.deepcopy(groups_winners[0]),int(n/8),clf.copy()),
        (candidates,copy.deepcopy(groups_pfs[0]),copy.deepcopy(groups_winners[0]),int(n/8),clf.copy()),
        (candidates,copy.deepcopy(groups_pfs[1]),copy.deepcopy(groups_winners[1]),int(n/8),clf.copy()),
        (candidates,copy.deepcopy(groups_pfs[1]),copy.deepcopy(groups_winners[1]),int(n/8),clf.copy()),
        (candidates,copy.deepcopy(groups_pfs[2]),copy.deepcopy(groups_winners[2]),int(n/8),clf.copy()),
        (candidates,copy.deepcopy(groups_pfs[2]),copy.deepcopy(groups_winners[2]),int(n/8),clf.copy()),
        (candidates,copy.deepcopy(groups_pfs[3]),copy.deepcopy(groups_winners[3]),int(n/8),clf.copy()),
        (candidates,copy.deepcopy(groups_pfs[3]),copy.deepcopy(groups_winners[3]),int(n/8),clf.copy()),
        ]
        L = pool.starmap(consistency_satisfaction_model_quick_process, parm_list)
        print(L)
        return (L[0]+L[1]+L[2]+L[3]+L[4]+L[5]+L[6]+L[7])/8    


def consistency_satisfaction_model_quick_process(candidates,group_pfs,group_winners,n,model_name):
    print("start model")
    sys.stdout.flush()
    time1 = time.time()
    pred_time = 0

    profile_count = 0
    consistency_count = 0
    new_pps = []
    while profile_count<n:
        pp1 = random.choice(group_pfs)
        pp2 = random.choice(group_pfs)
        new_pp = pp1+pp2
        pp1_Xs=np.array(labeling.get_Xs(candidates,[pp1]))
        pp2_Xs=np.array(labeling.get_Xs(candidates,[pp2]))
        new_pp_Xs=np.array(labeling.get_Xs(candidates,[new_pp]))
        t1 = time.time()
        preds_pp1 = group_winners[0]
        preds_newpp=np.argmax(clf.inplace_predict(new_pp_Xs),axis = 1)
        t2 = time.time()
        pred_time+=t2-t1

        if preds_pp1==preds_newpp:
           consistency_count+=1

    time2 = time.time()
    print(time2-time1,pred_time)
    return(consistency_count/profile_count)

def monotonicity_satisfaction_model(candidates,preference_profiles,n,model_name):
    
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    true = 0 
    for i in range(n):
        pf = random.choice(preference_profiles)
        old_Xs=np.array(labeling.get_Xs(candidates,[pf]))
        w =np.argmax(clf.predict(xgb.DMatrix(old_Xs)),axis = 1)[0]
        
        ballots = copy.deepcopy(preference_profile)
        for i,v in enumerate(ballots):
            w_idx = v.index(w)
            if(w_idx > 0):
                # if winner is not 1st choice in this vote
                #   swap winner with someone higher
                new_idx = random.randint(0,w_idx)
                temp = ballots[i][new_idx]
                ballots[i][new_idx] = ballots[i][w_idx]
                ballots[i][w_idx] = temp
        new_Xs = np.array(labeling.get_Xs(candidates,[ballots]))
        new_w = np.argmax(clf.predict(xgb.DMatrix(new_Xs)),axis = 1)[0]
        if w == new_w:
            true+=1
    return true/n

def generate_by_condorcet(candidates,voters,n):
    current_n = 0
    pfs = []
    winners = []
    while(current_n <= n):
        pf = data_generator.generate_samples(candidates,voters,1)[0]
        condorcet_winner = Condorcet(candidates,pf)
        if condorcet_winner != None:
            current_n += 1
            pfs.append(pf)
            winners.append(condorcet_winner)
    return pfs, winners
            
def generate_by_neutrality(candidates,pf,winner = None):
    perms = list(permutations(candidates))
    perms = [list(ele) for ele in perms] 
    if winner!=None:
        winner_index = candidates.index(winner)
    #print(perms)
    pfs = []
    winners = []
    for i in range(len(perms)):     
        perm = perms[i]
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
        pfs.append(copy.deepcopy(new_pf))
    if winner == None:
        return pfs
    else:
        return pfs,winners
    
def generate_by_consistency(candidates,pfs,labels,n):
    count = 0
    size = len(pfs)
    new_pfs = []
    new_labels = []
    while 1:
        if count == n:
            break
        i1 = random.randint(0,size-1)
        i2 = random.randint(0,size-1)
        if labels[i1] == labels[i2]:
            pf1 = pfs[i1]
            pf2 = pfs[i2]
            new_pf = pf1 + pf2
            new_pfs.append(new_pf)
            new_labels.append(labels[i1])
            #print(pfs[i1],pfs[i2],labels[i1],labels[i2],new_pf)
            count+=1

    return new_pfs, new_labels

def generate_by_consistency1(candidates,pfs1,labels1,pfs2,labels2,n):
    count = 0
    size1= len(pfs1)
    size2= len(pfs2)
    if size1 <50 and size2<50 :
        pfs = pfs1+pfs2
        labels = labels1+labels2
        return generate_by_consistency(candidates,pfs,labels,n)

    new_pfs = []
    new_labels = []
    while 1:
        if count == n:
            break
        i1 = random.randint(0,size1-1)
        i2 = random.randint(0,size2-1)
        if labels1[i1] == labels2[i2]:
            pf1 = pfs1[i1]
            pf2 = pfs2[i2]
            new_pf = pf1 + pf2
            new_pfs.append(new_pf)
            new_labels.append(labels1[i1])
            #print(pfs[i1],pfs[i2],labels[i1],labels[i2],new_pf)
            count+=1

    return new_pfs, new_labels
                           

def generate_by_monotonicity(candidates,preference_profile,label,n):
    w = label
    new_pfs = []
    winners = []
    for k in range(n):
        ballots = copy.deepcopy(preference_profile)
        for i,v in enumerate(ballots):
            w_idx = v.index(w)
            if(w_idx > 0):
                # if winner is not 1st choice in this vote
                #   swap winner with someone higher
                new_idx = random.randint(0,w_idx)
                temp = ballots[i][new_idx]
                ballots[i][new_idx] = ballots[i][w_idx]
                ballots[i][w_idx] = temp
        new_pfs.append(copy.deepcopy(ballots))
        winners.append(w)
    return new_pfs,winners

def choose_pf_by_order(candidates,pfs,model_name,n, labels = None):
    if (len(pfs) <n):
        if labels != None:
            return [],[],[]
        return [],[]
    if labels != None:
        new_pfs = copy.deepcopy(pfs[0:n])
        new_labels = copy.deepcopy(labels[0:n])
        del labels[0:n]
        del pfs[0:n]
        return new_pfs, new_labels, []
    else:
        new_pfs = copy.deepcopy(pfs[0:n])
        del pfs[0:n]
        return new_pfs,[]

def choose_pf_randomly(candidates,pfs,model_name,n, labels = None):
    if len(pfs)<n:
        if labels != None:
            return [],[],[]
        return [],[]
    if labels != None:
        new_pfs = []
        new_labels = []
        for i in range(n):
            random_index = random.randint(0,len(pfs)-1)
            new_pfs.append(copy.deepcopy(pfs[random_index]))
            new_labels.append(copy.deepcopy(labels[random_index]))
            pfs.pop(random_index)
            labels.pop(random_index)
        return new_pfs,new_labels,[]
    else:
        new_pfs = []
        for i in range(n):
            random_index = random.randint(0,len(pfs)-1)
            new_pfs.append(copy.deepcopy(pfs[random_index]))
            pfs.pop(random_index)
        return new_pfs,[]

    
def choose_pf_by_uncertainty(candidates,preference_profiles,model_name,n, labels = None):
    if (len(preference_profiles) == 0):
        if labels != None:
            return [],[],[]
   
        return [],[]

    pfs = []
    labs = []
    indexes = []
    for i in range(20):
        randindex = random.randrange(0,len(preference_profiles)-1)
        pfs.append(copy.deepcopy(preference_profiles[randindex]))
        indexes.append(randindex)
        if labels != None:
            labs.append(copy.deepcopy(labels[randindex]))
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    entropies = []
    Xs = np.array(labeling.get_Xs(candidates,pfs))
    preds = np.array(clf.predict(xgb.DMatrix(Xs)))
    #print("next preds:",preds)
    #print(preds)
    for pred in preds:
        entropies.append(np.sum(entr(pred)))
    entropies = np.array(entropies)
    #print(entropies)
    #print(entropies)
    best_indexs = np.argsort(entropies)[-n:]
    
    #print(max_indexs)
    max_pfs = [pfs[i] for i in best_indexs]
    max_indexes = [indexes[i] for i in best_indexs]
    if labels != None:
        max_labels = [labs[i] for i in best_indexs]
        return max_pfs, max_labels, max_indexes
    return max_pfs,max_indexes

def cross_entropy_loss(labels, predictions):
    loss = 0
    for i in range(len(labels)):
        loss-=math.log(predictions[i][labels[i]])
    return loss
#cross_entropy_loss([1,0],[[0.9,0.1],[0.1,0.9]])
#get_pool_mean([[1,2,3],[3,2,1]])
def generate_samples_random(candidates,voters,num_sample):
        preference_profiles = []
        n_voters = random.randint(1,len(voters))
        for i in range(num_sample):
            preference_profile = []
            for j in range(len(voters)):
                preference_profile.append(data_generator.random_ranking(candidates))
          
            preference_profiles.append(preference_profile)
        return preference_profiles
def generate_by_condorcet_random(candidates,voters,n):
    current_n = 0
    pfs = []
    winners = []
    while(current_n < n):
        pf = generate_samples_random(candidates,voters,1)[0]
        condorcet_winner = Condorcet(candidates,pf)
        if condorcet_winner != None:
            current_n += 1
            pfs.append(pf)
            winners.append(condorcet_winner)
    return pfs, winners

class CustomEnv(gym.Env):
  #"""Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}
  def __init__(self,designer,log):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.log = log
    dictionary = {'measurements':[],'final_states':[]}
    np.save(log + '/my_file.npy', dictionary) 
    self.max_n = 2000

    self.BUDGET = 100
    self.designer = designer
    #spaces.Box(low=0, high=255,shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low = -111111, high = 111111, shape=(state_space_dimension,),dtype=np.float32)
    """
    self.rule = designer
    if self.rule == "borda":
        self.voting_rule = borda
    if self.rule == "copland":
        self.voting_rule = copland
    if self.rule == "minimax":
        self.voting_rule = minimax
    if self.rule == "borda":
        self.labels_test = labels_test_borda
    if self.rule == "copland":
        self.labels_test = labels_test_copland
    if self.rule == "minimax":
        self.labels_test = labels_test_minimax
    self.neutrality_Xs = []
    self.neutrality_ys = []
    for i in range(len(Xs_test)):
        pfs,winners = generate_by_neutrality(candidates,preference_profiles_test[i],self.labels_test[i])
        print(len(self.neutrality_Xs))
        self.neutrality_Xs.extend(labeling.get_Xs(candidates,pfs))
        self.neutrality_ys.extend(winners)
    """

    self.reset()

  def seed(self,number):
    self.seed = number
  def step(self, action):
    #print("labeled_size:",len(self.train_pfs),",  stream_index:", self.stream_index)
    #print("self.state is: ",self.state)
    print("action:", action)
    done = False
    if action == 0:
        """
        if action == 0:
            new_pfs,indexes = self.choose_expert_pfs(1)
        else:
        """
        new_pfs,indexes = choose_pf_by_uncertainty(candidates,self.expert_unlabeled_pool_pfs,model_name,1, labels = None)
        new_labels = labeling.get_labels(candidates,new_pfs,self.voting_rule)
        self.expert_labeled_pool_pfs.extend(copy.deepcopy(new_pfs))
        self.expert_labeled_pool_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.expert_labeled_pool_labels.extend(new_labels)
        """
        for pf_index in range(len(new_pfs)):
            pfs,winners = generate_by_neutrality(candidates,new_pfs[pf_index],winner = new_labels[pf_index])
            self.neutrality_unlabeled_pool_pfs.extend(pfs)
            self.neutrality_unlabeled_pool_labels.extend(winners)
        """
        self.train_pfs.extend(copy.deepcopy(new_pfs))
        self.train_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.train_labels.extend(new_labels)
       
        active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
        self.number_of_expert_labels = len(self.expert_labeled_pool_pfs)
    
    if action == 1:
        n = 5
        if len(self.condorcet_labeled_pool_pfs) >= self.max_n:
            n = 1
        new_pfs, new_winners, indexes = self.choose_condorcet_pfs(n)
   
        self.condorcet_labeled_pool_pfs.extend(new_pfs)
        self.condorcet_labeled_pool_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.condorcet_labeled_pool_labels.extend(new_winners)
        
        self.train_pfs.extend(new_pfs)
        self.train_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.train_labels.extend(new_winners)
        active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
        self.number_of_condorcet_labels = len(self.condorcet_labeled_pool_pfs)
    if action == 2:
        n = 5
        if len(self.consistency_labeled_pool_pfs1)>self.max_n:
            n = 1
        composite_set = copy.deepcopy(self.expert_labeled_pool_pfs+self.condorcet_labeled_pool_pfs)
        composite_ys = copy.deepcopy(self.expert_labeled_pool_labels+self.condorcet_labeled_pool_labels)

        new_pfs,new_winners = generate_by_consistency(candidates,composite_set,composite_ys,n)
        #new_pfs, new_winners, indexes = self.choose_neutrality_pfs(20)
        #self.neutrality_labeled_pool_pfs.extend(new_pfs)
        #self.neutrality_labeled_pool_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        #self.neutrality_labeled_pool_labels.extend(new_winners)
        self.consistency_labeled_pool_pfs1.extend(new_pfs)
        self.consistency_labeled_pool_Xs1.extend(labeling.get_Xs(candidates,new_pfs))
        self.consistency_labeled_pool_labels1.extend(new_winners)
        
        self.train_pfs.extend(new_pfs)
        self.train_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.train_labels.extend(new_winners)

        active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
        #self.number_of_neutrality_labels = len(self.neutrality_labeled_pool_pfs)
        self.number_of_consistency_labels1 = len(self.consistency_labeled_pool_pfs1)
    if action == 3:
        n =5
        if len(self.consistency_labeled_pool_pfs2)>self.max_n:
            n = 1

        new_pfs,new_winners = generate_by_consistency1(candidates,self.expert_labeled_pool_pfs,self.expert_labeled_pool_labels,self.condorcet_labeled_pool_pfs,self.condorcet_labeled_pool_labels,n)
        #new_pfs, new_winners, indexes = self.choose_neutrality_pfs(20)
        #self.neutrality_labeled_pool_pfs.extend(new_pfs)
        #self.neutrality_labeled_pool_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        #self.neutrality_labeled_pool_labels.extend(new_winners)
        self.consistency_labeled_pool_pfs2.extend(new_pfs)
        self.consistency_labeled_pool_Xs2.extend(labeling.get_Xs(candidates,new_pfs))
        self.consistency_labeled_pool_labels2.extend(new_winners)
        
        self.train_pfs.extend(new_pfs)
        self.train_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.train_labels.extend(new_winners)

        active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
        #self.number_of_neutrality_labels = len(self.neutrality_labeled_pool_pfs)
        self.number_of_consistency_labels2 = len(self.consistency_labeled_pool_pfs2)
    
    self.state = self.get_state(model_name)
    done = (len(self.expert_labeled_pool_pfs) >= self.BUDGET)
        #+len(self.condorcet_labeled_pool_pfs)*0.01+len(self.consistency_labeled_pool_pfs2)*0.01+len(self.consistency_labeled_pool_pfs1)*0.01>= self.BUDGET) or (len(self.condorcet_labeled_pool_pfs) >=3000) or (len(self.consistency_labeled_pool_pfs1) >=3000) or (len(self.consistency_labeled_pool_pfs2) >=3000)
    #return observation, reward, done, info

    if done == True:
        """
        new_test_X = copy.deepcopy(Xs_test)
        new_test_y = copy.deepcopy(self.labels_test)
        neutrality_n = int(self.number_of_neutrality_labels/self.number_of_expert_labels*len(Xs_test))
        condorcet_n = int(self.number_of_condorcet_labels/self.number_of_expert_labels*len(Xs_test))
        for j in range(neutrality_n):
            rand_indx = random.randint(0,len(self.neutrality_Xs)-1)
            new_test_X.append(self.neutrality_Xs[rand_indx])
            new_test_y.append(self.neutrality_ys[rand_indx])

        for j in range(condorcet_n):
            rand_indx = random.randint(0,len(preference_profiles_condorcet_Xs_test)-1)
            new_test_X.append(preference_profiles_condorcet_Xs_test[rand_indx])
            new_test_y.append(labels_condorcet_test[rand_indx])
        self.test_accuracy = active_learning.prediction_accuracy(np.array(new_test_X),np.array(new_test_y),model_name)
        """
        self.designer_accuracy = active_learning.prediction_accuracy(np.array(Xs_test),np.array(self.labels_test),model_name)
        #self.neutrality_satisfaction = neutrality_satisfaction_model(candidates,preference_profiles_test,20000,model_name)
        self.consistency_satisfaction = consistency_satisfaction_model_quick(candidates,preference_profiles_test,20000,model_name)
        self.condorcet_satisfaction = condorcet_satisfaction(np.array(preference_profiles_condorcet_Xs_test),np.array(labels_condorcet_test),model_name)
        self.reward = (2*self.designer_accuracy+self.condorcet_satisfaction+self.consistency_satisfaction)/4

        # Save


        # Load
        read_dictionary = copy.deepcopy(np.load(self.log+'/my_file.npy',allow_pickle='TRUE').item())
        print(read_dictionary['measurements'][0:min(5,len(read_dictionary['measurements']))])
        (read_dictionary['measurements']).append((self.designer_accuracy,self.consistency_satisfaction,self.condorcet_satisfaction)) # displays "world"
        (read_dictionary['final_states']).append(tuple(self.state))
        np.save(self.log+'/my_file.npy', read_dictionary) 
        """
        if self.test_accuracy<0.95:
            self.reward = self.reward + self.test_accuracy-0.95
        """
        print(self.designer_accuracy,self.consistency_satisfaction,self.condorcet_satisfaction,self.reward)
        reward = self.reward
    else:
        reward = 0
    self.print_status()
    return np.array(self.state), reward, done, dict()
  def reset(self):
    print("reset")
    #warm_up_size = random.randint(40,100)
    if self.designer == "random":
        self.rule = random.choice(["borda","copland","minimax"])
    if self.designer == "plurality":
        self.rule = "plurality"
    if self.designer == "borda":
        self.rule = "borda"
        
    if self.rule == "borda":
        self.voting_rule = borda
    if self.rule == "copland":
        self.voting_rule = copland
    if self.rule == "minimax":
        self.voting_rule = minimax
    if self.rule == "plurality":
        self.voting_rule = plurality
    if self.rule == "borda":
        self.labels_test = labels_test_borda
    if self.rule == "copland":
        self.labels_test = labels_test_copland
    if self.rule == "minimax":
        self.labels_test = labels_test_minimax
    if self.rule == "plurality":
        self.labels_test = labels_test_plurality
    
    self.pfs_warm_up = [copy.deepcopy(preference_profiles[150])]

    self.expert_unlabeled_pool_pfs = copy.deepcopy(preference_profiles)
    self.expert_labeled_pool_pfs = copy.deepcopy(self.pfs_warm_up)
    self.expert_labeled_pool_Xs = labeling.get_Xs(candidates,self.pfs_warm_up)
    self.expert_labeled_pool_labels = copy.deepcopy(labeling.get_labels(candidates,self.pfs_warm_up,self.voting_rule))
    
    self.condorcet_unlabeled_pool_pfs = copy.deepcopy(preference_profiles_condorcet)
    self.condorcet_unlabeled_pool_labels = copy.deepcopy(labels_condorcet)
    self.condorcet_labeled_pool_pfs = []
    self.condorcet_labeled_pool_Xs = []
    self.condorcet_labeled_pool_labels = []
    """
    self.neutrality_unlabeled_pool_pfs = []
    self.neutrality_unlabeled_pool_labels = []
    for pf_index in range(len(self.expert_labeled_pool_pfs)):
        pfs,winners = generate_by_neutrality(candidates,self.expert_labeled_pool_pfs[pf_index],winner = self.expert_labeled_pool_labels[pf_index])
        self.neutrality_unlabeled_pool_pfs.extend(pfs)
        self.neutrality_unlabeled_pool_labels.extend(winners)
    self.neutrality_labeled_pool_pfs = []
    self.neutrality_labeled_pool_Xs = []
    self.neutrality_labeled_pool_labels = []
    """
    self.consistency_labeled_pool_pfs1 = []
    self.consistency_labeled_pool_Xs1 = []
    self.consistency_labeled_pool_labels1 = []

    self.consistency_labeled_pool_pfs2 = []
    self.consistency_labeled_pool_Xs2 = []
    self.consistency_labeled_pool_labels2 = []
    
    self.train_pfs = copy.deepcopy(self.pfs_warm_up)
    self.train_Xs = labeling.get_Xs(candidates,self.train_pfs)
    self.train_labels = labeling.get_labels(candidates,self.pfs_warm_up,self.voting_rule)
    active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
    
    self.reward = 0

    self.state = self.get_state(model_name)
    self.print_status()
    return np.array(self.state)  # reward, done, info can't be included
  def loss(self,Xs, labels,model_name):
    if (len(labels) == 0): return 0
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    preds = clf.predict(xgb.DMatrix(Xs))
    preds = np.array(preds)
    loss = cross_entropy_loss(labels,preds)/len(labels)*1000
    #preds = np.argmax(preds,axis = 1)
    #error = np.mean( preds != labels )
    #print("percentage Error:",error)
    return loss
    
  def choose_expert_pfs(self,n):
    return choose_pf_randomly(candidates,self.expert_unlabeled_pool_pfs,model_name,n)

  def choose_condorcet_pfs(self,n):
    """
    pfs = []
    labels = []
    for i in range(n):

        random_index = random.randint(0, len(preference_profiles_condorcet)-1)
        pfs.append(copy.deepcopy(preference_profiles_condorcet[random_index]))
        labels.append(copy.deepcopy(labels_condorcet[random_index]))
    return pfs,labels
    """
    return choose_pf_randomly(candidates,self.condorcet_unlabeled_pool_pfs,model_name,n,self.condorcet_unlabeled_pool_labels)
 
  def choose_neutrality_pfs(self,n):
    return choose_pf_randomly(candidates,self.neutrality_unlabeled_pool_pfs,model_name,n,self.neutrality_unlabeled_pool_labels)
 
  def get_state(self, model_name):
    l = []
    l.append(len(self.expert_labeled_pool_pfs))
    l.append(len(self.condorcet_labeled_pool_pfs))
    #l.append(len(self.neutrality_labeled_pool_pfs))
    #l.append(len(self.expert_unlabeled_pool_pfs))
    #l.append(len(self.condorcet_unlabeled_pool_pfs))
   # l.append(len(self.neutrality_unlabeled_pool_pfs))
    l.append(len(self.consistency_labeled_pool_pfs1))
    l.append(len(self.consistency_labeled_pool_pfs2))
    """
    train_X = []
    train_y = []
    train_X.extend(self.expert_labeled_pool_Xs)
    train_X.extend(self.condorcet_labeled_pool_Xs)
    train_X.extend(self.consistency_labeled_pool_Xs1)
    train_X.extend(self.consistency_labeled_pool_Xs2)
    #train_X.extend(self.neutrality_labeled_pool_Xs)
    train_y.extend(self.expert_labeled_pool_labels)
    train_y.extend(self.condorcet_labeled_pool_labels)
    #train_y.extend(self.neutrality_labeled_pool_labels)
    train_y.extend(self.consistency_labeled_pool_labels1)
    train_y.extend(self.consistency_labeled_pool_labels2)
    total_loss = (self.loss(np.array(train_X), np.array(train_y),model_name))
    """
    expert_loss = self.loss(np.array(self.expert_labeled_pool_Xs), np.array(self.expert_labeled_pool_labels),model_name)
    condorcet_loss = self.loss(np.array(self.condorcet_labeled_pool_Xs), np.array(self.condorcet_labeled_pool_labels),model_name)
    consistency_loss1 = self.loss(np.array(self.consistency_labeled_pool_Xs1), np.array(self.consistency_labeled_pool_labels1),model_name)
    consistency_loss2 = self.loss(np.array(self.consistency_labeled_pool_Xs2), np.array(self.consistency_labeled_pool_labels2),model_name)
    #l.append((expert_loss*len(self.expert_labeled_pool_Xs)+condorcet_loss*(len(self.condorcet_labeled_pool_pfs))+consistency_loss1*(len(self.consistency_labeled_pool_pfs1))+consistency_loss2*(len(self.consistency_labeled_pool_pfs2)))/len(self.train_pfs)*1000)
    l.append(expert_loss)
    l.append(condorcet_loss)
    l.append(consistency_loss1)
    l.append(consistency_loss2)

    #X_size = len(self.expert_labeled_pool_Xs[0])
    #l.append(get_pool_mean(self.expert_labeled_pool_Xs,X_size))
    #l.append(get_pool_mean(self.condorcet_labeled_pool_Xs,X_size))
    #l.append(get_pool_mean(self.neutrality_labeled_pool_Xs,X_size))
    return l
 
  def print_status(self):
    print(self.rule)
    print(self.state)
    #print(self.borda_accuracy,self.condorcet_satisfaction)
    #print("self.state is: ",self.state)
    #print(active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name))

  def render(self):  #mode='human'
    return None
  def close (self):
    return None
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        return

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-20:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model

                    self.save_path = os.path.join(log_dir, 'model'+str(self.n_calls))
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    if self.save_path is not None:
                        os.makedirs(self.save_path, exist_ok=True)

                    self.model.save(self.save_path)

        return True
if __name__=='__main__':
    model_name = "model4"
    if os.path.isfile(model_name):
       os.remove(model_name)
    plurality = voting_rules.voting_rule0A
    copland  = voting_rules.voting_rule2A
    borda = voting_rules.voting_rule0
    minimax = voting_rules.voting_rule3
    n_candidates = 3
    n_voters = 5

    candidates = list(range(0,n_candidates))
    voters = ['1']*n_voters

    preference_profiles = generate_samples_random(candidates,voters,10000)
    preference_profiles_test = generate_samples_random(candidates,voters,30000)
    Xs_train = labeling.get_Xs(candidates,preference_profiles)
    Xs_test = labeling.get_Xs(candidates,preference_profiles_test)
    labels_test_plurality = labeling.get_labels(candidates,preference_profiles_test,plurality)
    labels_train_plurality = labeling.get_labels(candidates,preference_profiles,plurality)
    labels_test_borda = labeling.get_labels(candidates,preference_profiles_test,borda)
    labels_train_borda = labeling.get_labels(candidates,preference_profiles,borda)
    labels_test_copland = labeling.get_labels(candidates,preference_profiles_test,copland)
    labels_train_copland = labeling.get_labels(candidates,preference_profiles,copland)
    labels_test_minimax = labeling.get_labels(candidates,preference_profiles_test,minimax)
    labels_train_minimax = labeling.get_labels(candidates,preference_profiles,minimax)


    preference_profiles_condorcet, labels_condorcet = generate_by_condorcet_random(candidates,voters,30000)
    preference_profiles_condorcet_test, labels_condorcet_test = generate_by_condorcet_random(candidates,voters,30000)
    preference_profiles_condorcet_Xs = labeling.get_Xs(candidates,preference_profiles_condorcet)
    preference_profiles_condorcet_Xs_test = labeling.get_Xs(candidates,preference_profiles_condorcet_test)

    params = {'objective':'multi:softprob', 'max_depth' :3,'n_estimators':3, 'num_class':len(candidates)}#WAS 10,5
    import multiprocessing as mp
    import time
    print(mp.cpu_count())

    anchor_number = 100
    anchor_pfs = preference_profiles[0:anchor_number]

    state_space_dimension = 8
    #state_space_dimension = 10*15*3+3+15+1
    #state_space_dimension = (n_candidates) * (anchor_number+1)+1
    #state_space_dimension = (anchor_number+1)+1
    #(anchor_number+1)+1
    N_DISCRETE_ACTIONS = 4
    initial_state = 0

    #state_space_dimension = (9+6+1) * (BUDGET+2)-1
    #state_space_dimension = (3) * (BUDGET+2)
    #env = make_vec_env(CustomEnv("borda"), n_envs=1)
    log_dir = "consistency_PPO2_10BUDGET_0/"
    import shutil
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    env = CustomEnv("borda",log_dir)
    env.BUDGET = 10
    env = Monitor(env, log_dir)
    #env = SubprocVecEnv([make_env("borda","consistency/", i) for i in range(4)])
    env = DummyVecEnv([lambda: env])

    #env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_obs=10.)

    #env = DummyVecEnv([Monitor(env, log_dir)])
    #env = make_vec_env('CartPole-v1', n_envs=1)
    call_back = SaveOnBestTrainingRewardCallback(check_freq=20000, log_dir=log_dir)
    #env = SubprocVecEnv([env])
    #model = PPO2.load("consistency_PPO2_10BUDGET_A",env = env,verbose = 1)
    #model = TRPO(MlpPolicy, env, gamma = 1,verbose=1)
    #model = DQN(CustomDQNPolicy, env, gamma=0.999, verbose=1)
    #policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256,256,128,128,64])
    #model = A2C(MlpLstmPolicy,env, gamma = 1,verbose=1)
    model = PPO2(MlpPolicy, env, learning_rate=0.000025,gamma = 0.999,verbose=1, nminibatches=1,noptepochs = 16,n_steps = 700)
    #model.set_env(env)
    model.learn(300000,callback = call_back)
    #stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    #env.save(stats_path)

    model.save("consistency_PPO2_10BUDGET_A")
