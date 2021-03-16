#!/usr/bin/env python
# coding: utf-8

# In[279]:


import numpy as np
import random
import math
from IPython.display import clear_output
from collections import deque
import progressbar

import gym
from gym import spaces
from stable_baselines.common.env_checker import check_env

import tensorflow.keras as keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

from scipy.special import entr


# In[280]:


import os
print(os.getcwd())
os.chdir(r"C:\Users\wangy45\Documents\Research\new")
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
random.seed(30)


# In[298]:


from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import LstmPolicy

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines import PPO2
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
import tensorflow as tf


# In[314]:


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
def choose_pf_by_order(candidates,preference_profiles,model_name,n, labels = None):
    if (len(preference_profiles) <n):
        if labels != None:
            return [],[],[]
        return [],[]
    if labels != None:
        new_pfs = copy.deepcopy(preference_profiles[0:n])
        new_labels = copy.deepcopy(labels[0:n])
        indexs = []
        if len(preference_profiles) >2*n:
            del labels[0:n]
            del preference_profiles[0:n]
        return new_pfs, new_labels, indexs
    else:
        new_pfs = copy.deepcopy(preference_profiles[0:n])
        
        if len(preference_profiles) >2*n: preference_profiles[0:n]
        return new_pfs,[]
    
def choose_pf_by_uncertainty(candidates,preference_profiles,model_name,n, labels = None):
    if (len(preference_profiles) == 0):
        if labels != None:
            return [],[],[]
   
        return [],[]

    pfs = []
    labs = []
    for i in range(20):
        randindex = random.randrange(0,len(preference_profiles)-1)
        pfs.append(copy.deepcopy(preference_profiles[randindex]))
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
    max_indexs = np.argsort(entropies)[-n:]
    
    #print(max_indexs)
    max_pfs = [pfs[i] for i in max_indexs]
    if labels != None:
        max_labels = [labs[i] for i in max_indexs]
        return max_pfs, max_labels, max_indexs
    return max_pfs,max_indexs

def cross_entropy_loss(labels, predictions):
    loss = 0
    for i in range(len(labels)):
        loss-=math.log(predictions[i][labels[i]])
    return loss/len(labels)
#cross_entropy_loss([1,0],[[0.9,0.1],[0.1,0.9]])
#get_pool_mean([[1,2,3],[3,2,1]])


# In[283]:


"""
def prediction_accuracy(Xs,true_ys,clf):
    preds = clf.predict(xgb.DMatrix(Xs))
    preds = np.array(preds)
    preds = np.argmax(preds,axis = 1)
    error = np.mean( preds != true_ys )
    print("percentage Error:",error)
    return 1-error

def initialize_model(new_Xs,new_ys,params,model_name):
    data = xgb.DMatrix(new_Xs, label=new_ys)
    #clf = xgb.train(params,data)
    clf = xgb.train(params,data,xgb_model=model_name)
    clf.save_model(model_name)
    #return clf
def bin_array(num, m):
    #Convert a positive integer num into an m-bit bit vector
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.float32)
"""


# In[284]:


model_name = "model3"
if os.path.isfile(model_name):
   os.remove(model_name)
voting_rule  = voting_rules.voting_rule0
#voting_rule = voting_rules.voting_rule0
n_candidates = 4
n_voters = 20

candidates = list(range(0,n_candidates))
voters = ['1']*n_voters
params = {'objective':'multi:softprob', 'max_depth' :15,'n_estimators':40, 'num_class':len(candidates)}
preference_profiles = data_generator.generate_samples(candidates,voters,200)
preference_profiles_test = data_generator.generate_samples(candidates,voters,30000)
Xs_train = labeling.get_Xs(candidates,preference_profiles)
Xs_test = labeling.get_Xs(candidates,preference_profiles_test)
labels_test = labeling.get_labels(candidates,preference_profiles_test,voting_rule)
labels_train = labeling.get_labels(candidates,preference_profiles,voting_rule)

preference_profiles_condorcet, labels_condorcet = generate_by_condorcet(candidates,voters,3000)
preference_profiles_condorcet_test, labels_condorcet_test = generate_by_condorcet(candidates,voters,30000)
preference_profiles_condorcet_Xs = labeling.get_Xs(candidates,preference_profiles_condorcet)
preference_profiles_condorcet_Xs_test = labeling.get_Xs(candidates,preference_profiles_condorcet_test)

train_Xs = []
train_labels = []
train_Xs.extend(preference_profiles_condorcet_Xs[0:3000])
train_Xs.extend(Xs_train[0:1000])
train_labels.extend(labels_condorcet[0:3000])
train_labels.extend(labels_train[0:1000])
active_learning.initialize_model(np.array(train_Xs),np.array(train_labels),params,model_name)
choose_pf_by_uncertainty(candidates,preference_profiles[0:10],model_name, 5,labels = None)
#print(condorcet_satisfaction(np.array(preference_profiles_condorcet_Xs_test),np.array(labels_condorcet_test),model_name))
#print(active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name))
#print(condorcet_satisfaction(candidates,preference_profiles_condorcet_test,labels_condorcet_test,voting_rule,model_name = None))
#print(get_X_histograms_long(Xs_test,labels_test,n_voters))
#print(get_y_histograms(labels_test,n_candidates))


# In[285]:


anchor_number = 100
anchor_pfs = preference_profiles[0:anchor_number]

state_space_dimension = 13
#state_space_dimension = 10*15*3+3+15+1
#state_space_dimension = (n_candidates) * (anchor_number+1)+1
#state_space_dimension = (anchor_number+1)+1
#(anchor_number+1)+1


# In[286]:


N_DISCRETE_ACTIONS = 3
initial_state = 0

#state_space_dimension = (9+6+1) * (BUDGET+2)-1
#state_space_dimension = (3) * (BUDGET+2)


# In[315]:


class CustomEnv(gym.Env):
  #"""Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}
  def __init__(self):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.BUDGET = 200
    #spaces.Box(low=0, high=255,shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low = -111111, high = 111111, shape=(state_space_dimension,),dtype=np.float32)
    self.reset()
  def step(self, action):
    #print("labeled_size:",len(self.train_pfs),",  stream_index:", self.stream_index)
    #print("self.state is: ",self.state)

    self.accuracy_old = self.accuracy_new
    if action == 0:
        new_pfs,indexes = self.choose_expert_pfs(10)
        new_labels = labeling.get_labels(candidates,new_pfs,voting_rule)
       
        self.expert_labeled_pool_pfs.extend(copy.deepcopy(new_pfs))
        self.expert_labeled_pool_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.expert_labeled_pool_labels.extend(new_labels)
        for pf_index in range(len(new_pfs)):
            pfs,winners = generate_by_neutrality(candidates,new_pfs[pf_index],winner = new_labels[pf_index])
            self.neutrality_unlabeled_pool_pfs.extend(pfs)
            self.neutrality_unlabeled_pool_labels.extend(winners)
        
        self.train_pfs.extend(copy.deepcopy(new_pfs))
        self.train_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.train_labels.extend(new_labels)
       
        active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
        self.number_of_expert_labels = len(self.expert_labeled_pool_pfs)
    if action == 1:
        new_pfs, new_winners, indexes = self.choose_condorcet_pfs(100)
        
        self.condorcet_labeled_pool_pfs.extend(new_pfs)
        self.condorcet_labeled_pool_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.condorcet_labeled_pool_labels.extend(new_winners)
        
        self.train_pfs.extend(new_pfs)
        self.train_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.train_labels.extend(new_winners)
        active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
        self.number_of_condorcet_labels = len(self.condorcet_labeled_pool_pfs)
    if action == 2:
        new_pfs, new_winners, indexes = self.choose_neutrality_pfs(100)
        self.neutrality_labeled_pool_pfs.extend(new_pfs)
        self.neutrality_labeled_pool_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.neutrality_labeled_pool_labels.extend(new_winners)
        
        self.train_pfs.extend(new_pfs)
        self.train_Xs.extend(labeling.get_Xs(candidates,new_pfs))
        self.train_labels.extend(new_winners)
        active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
        self.number_of_neutrality_labels = len(self.neutrality_labeled_pool_pfs)
        
    self.borda_accuracy = active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name)
    self.condorcet_satisfaction = condorcet_satisfaction(np.array(preference_profiles_condorcet_Xs_test),np.array(labels_condorcet_test),model_name)
    self.accuracy_new = (self.borda_accuracy+self.condorcet_satisfaction)/2
    
    self.state = self.get_state(model_name)
    done = len(self.expert_labeled_pool_pfs) + len(self.condorcet_labeled_pool_pfs)*0.02 + len(self.neutrality_labeled_pool_pfs)*0.02>= self.BUDGET
    #return observation, reward, done, info
    reward = self.accuracy_new-self.accuracy_old
    self.print_status()
    return np.array(self.state), reward, done, dict()
  def reset(self):
    #warm_up_size = random.randint(40,100)
    
    
    self.pfs_warm_up = [copy.deepcopy(preference_profiles[150])]

    self.expert_unlabeled_pool_pfs = copy.deepcopy(preference_profiles)
    self.expert_labeled_pool_pfs = copy.deepcopy(self.pfs_warm_up)
    self.expert_labeled_pool_Xs = labeling.get_Xs(candidates,self.pfs_warm_up)
    self.expert_labeled_pool_labels = copy.deepcopy(labeling.get_labels(candidates,self.pfs_warm_up,voting_rule))
    
    self.condorcet_unlabeled_pool_pfs = copy.deepcopy(preference_profiles_condorcet)
    self.condorcet_unlabeled_pool_labels = copy.deepcopy(labels_condorcet)
    self.condorcet_labeled_pool_pfs = []
    self.condorcet_labeled_pool_Xs = []
    self.condorcet_labeled_pool_labels = []
    
    self.neutrality_unlabeled_pool_pfs = []
    self.neutrality_unlabeled_pool_labels = []
    for pf_index in range(len(self.expert_labeled_pool_pfs)):
        pfs,winners = generate_by_neutrality(candidates,self.expert_labeled_pool_pfs[pf_index],winner = self.expert_labeled_pool_labels[pf_index])
        self.neutrality_unlabeled_pool_pfs.extend(pfs)
        self.neutrality_unlabeled_pool_labels.extend(winners)
    self.neutrality_labeled_pool_pfs = []
    self.neutrality_labeled_pool_Xs = []
    self.neutrality_labeled_pool_labels = []
    
    self.train_pfs = copy.deepcopy(self.pfs_warm_up)
    self.train_Xs = labeling.get_Xs(candidates,self.train_pfs)
    self.train_labels = labeling.get_labels(candidates,self.pfs_warm_up,voting_rule)
    active_learning.initialize_model(np.array(self.train_Xs),np.array(self.train_labels),params,model_name)
    
    self.borda_accuracy = active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name)
    self.condorcet_satisfaction = condorcet_satisfaction(np.array(preference_profiles_condorcet_Xs_test),np.array(labels_condorcet_test),model_name)
    self.accuracy_new = (self.borda_accuracy+self.condorcet_satisfaction)/2
    self.state = self.get_state(model_name)
    self.print_status()
    return np.array(self.state)  # reward, done, info can't be included
  def loss(self,Xs, labels,model_name):
    if (len(labels) == 0): return 0
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    preds = clf.predict(xgb.DMatrix(Xs))
    preds = np.array(preds)
    loss = cross_entropy_loss(labels,preds)
    #preds = np.argmax(preds,axis = 1)
    #error = np.mean( preds != labels )
    #print("percentage Error:",error)
    return loss
    
  def choose_expert_pfs(self,n):
    return choose_pf_by_order(candidates,self.expert_unlabeled_pool_pfs,model_name,n)
    """
    pfs = []
    for i in range(n):
        pfs.append(random.choice(preference_profiles))
    return pfs
    """
    """
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    choices = []
    accuracies = []
    max_accuracy = 0
    max_choice = None
    for i in range(10):
        choices.append(copy.deepcopy(random.choice(preference_profiles)))
    for choice in choices:
        train_pfs = copy.deepcopy(self.train_pfs)
        train_Xs = copy.deepcopy(self.train_Xs)
        train_labels = copy.deepcopy(self.train_labels)
        new_Xs = labeling.get_Xs(candidates,[choice])
        preds = clf.predict(xgb.DMatrix(new_Xs))
        preds = np.array(preds)
        preds = np.argmax(preds,axis = 1)
        new_label = preds[0]
        train_pfs.append(choice)
        train_Xs.extend(new_Xs)
        train_labels.append(new_label)
        active_learning.initialize_model(np.array(train_Xs),np.array(train_labels),params,model_name)
        accuracy_new = active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name)
        accuracies.append(accuracy_new)
        if accuracy_new > max_accuracy:
            max_accuracy = accuracy_new
            max_choice = choice
    print(accuracies)
    return copy.deepcopy(max_choice)
    """
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
    return choose_pf_by_order(candidates,self.condorcet_unlabeled_pool_pfs,model_name,n,self.condorcet_unlabeled_pool_labels)
    """
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    choices = []
    labels =[]
    accuracies = []
    max_accuracy = 0
    max_choice = None
    max_label = None
    for i in range(10):
        random_index = random.randint(0, len(preference_profiles_condorcet)-1)
        choices.append(copy.deepcopy(preference_profiles_condorcet[random_index]))
        labels.append(copy.deepcopy(labels_condorcet[random_index]))
    for i in range(10):
        choice = choices[i]
        label = labels[i]
        train_pfs = copy.deepcopy(self.train_pfs)
        train_Xs = copy.deepcopy(self.train_Xs)
        train_labels = copy.deepcopy(self.train_labels)
        new_Xs = labeling.get_Xs(candidates,[choice])
        new_label = labels_condorcet
        train_pfs.append(choice)
        train_Xs.extend(new_Xs)
        train_labels.append(label)
        active_learning.initialize_model(np.array(train_Xs),np.array(train_labels),params,model_name)
        accuracy_new = active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name)
        accuracies.append(accuracy_new)
        if accuracy_new > max_accuracy:
            max_accuracy = accuracy_new
            max_choice = choice
            max_label =label
    print(accuracies)
    return copy.deepcopy(max_choice), copy.deepcopy(max_label)
    """
  def choose_neutrality_pfs(self,n):
    return choose_pf_by_order(candidates,self.neutrality_unlabeled_pool_pfs,model_name,n,self.neutrality_unlabeled_pool_labels)
    """
    pfs = []
    labels = []
    for i in range(10):
        pf_index = random.randint(0,len(self.expert_labeled_pfs)-1)
        pf,winner =generate_by_neutrality(candidates,self.expert_labeled_pfs[pf_index],1,winner = self.expert_labels[pf_index])
        pfs.append(pf[0])
        labels.append(winner[0])
    return pfs,labels
    """
    """
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    choices = []
    labels = []
    accuracies = []
    max_accuracy = 0
    max_choice = None
    max_winner = None
    for i in range(10):
        pf_index = random.randint(0,len(self.expert_labeled_pfs)-1)
        pf,winner =generate_by_neutrality(candidates,self.expert_labeled_pfs[pf_index],1,winner = self.expert_labels[pf_index])
        choices.append(pf[0])
        labels.append(winner[0])
    for i in range(10):
        choice = choices[i]
        label = labels[i]
        train_pfs = copy.deepcopy(self.train_pfs)
        train_Xs = copy.deepcopy(self.train_Xs)
        train_labels = copy.deepcopy(self.train_labels)
        new_Xs = labeling.get_Xs(candidates,[choice])
        train_pfs.append(choice)
        train_Xs.extend(new_Xs)
        train_labels.append(label)
        active_learning.initialize_model(np.array(train_Xs),np.array(train_labels),params,model_name)
        accuracy_new = active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name)
        accuracies.append(accuracy_new)
        if accuracy_new > max_accuracy:
            max_accuracy = accuracy_new
            max_choice = choice
            max_winner = label
    print(accuracies)
    
    return max_choice, max_winner
    """
  def get_state(self, model_name):
    l = []
    l.append(len(self.expert_labeled_pool_pfs))
    l.append(len(self.condorcet_labeled_pool_pfs))
    l.append(len(self.neutrality_labeled_pool_pfs))
    l.append(len(self.expert_unlabeled_pool_pfs))
    l.append(len(self.condorcet_unlabeled_pool_pfs))
    l.append(len(self.neutrality_unlabeled_pool_pfs))
     
    l.append(self.borda_accuracy)
    l.append(self.loss(np.array(Xs_test), np.array(labels_test),model_name))
    l.append(self.condorcet_satisfaction)
    Xs = np.array(labeling.get_Xs(candidates,copy.deepcopy(anchor_pfs)))
    #print("next Xs:",Xs)
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    preds = np.array(clf.predict(xgb.DMatrix(Xs)))
    #print("next preds:",preds)
    #print(preds)
    l.append(np.sum(entr(preds))/anchor_number)
    
    l.append(self.loss(np.array(self.expert_labeled_pool_Xs), np.array(self.expert_labeled_pool_labels),model_name))
    l.append(self.loss(np.array(self.condorcet_labeled_pool_Xs), np.array(self.condorcet_labeled_pool_labels),model_name))
    l.append(self.loss(np.array(self.neutrality_labeled_pool_Xs),np.array(self.neutrality_labeled_pool_labels),model_name))
    X_size = len(self.expert_labeled_pool_Xs[0])
    #l.append(get_pool_mean(self.expert_labeled_pool_Xs,X_size))
    #l.append(get_pool_mean(self.condorcet_labeled_pool_Xs,X_size))
    #l.append(get_pool_mean(self.neutrality_labeled_pool_Xs,X_size))
    return l
    """
    X = np.array(labeling.get_Xs(candidates,[next_pf]))[0]
    l = []
    l.extend(get_X_histograms_long(self.train_Xs,self.train_labels,n_voters).tolist())
    #print(len(l))
    l.extend((get_y_histograms(self.train_labels,n_candidates)).tolist())
    #print(len(l))
    l.extend(X)
    l.extend([self.number_of_labels])
    #print(len(l))
    return l
    """
    """
    #pfs = []
    pfs = copy.deepcopy(anchor_pfs)
    #pfs.append(next_pf)
    Xs = np.array(labeling.get_Xs(candidates,pfs))
    #print("next Xs:",Xs)
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(model_name)  # load data
    preds = np.array(clf.predict(xgb.DMatrix(Xs)))
    #print("next preds:",preds)
    #print(preds)
    
    rep = [np.sum(entr(preds))/anchor_number]
    
    Xs = np.array(labeling.get_Xs(candidates,[next_pf]))
    #print(rep)
    #print(preds)
    #preds = np.concatenate(preds).ravel()
    
    preds = np.array(clf.predict(xgb.DMatrix(Xs)))
    rep = np.append(rep,np.sum(entr(preds)))
    rep = np.append(rep,self.number_of_labels/BUDGET)
    #print(rep)
    return rep
    """
    
    """
    next_X = labeling.get_Xs(candidates,[next_pf])[0]
    pfs = []
    for i in range(len(self.train_Xs)):
        pfs.extend(self.train_Xs[i])
        pfs.append(self.train_labels[i])
    pfs = np.array(pfs)
    #pfs = np.concatenate(np.array(pfs)).ravel()
    #print(pfs)
    #print(next_X)
    rep = np.full((state_space_dimension,),-1)
    
    rep[0:pfs.shape[0],] =pfs 
    rep[np.shape(rep)[0]-(len(next_X)):np.shape(rep)[0]] = np.array(np.array(next_X))
    #print(np.shape(rep))
    #print(state_space_dimension)
    #print(rep[0:20])
    return rep
    """
  def print_status(self):
    print(self.state)
    #print("self.state is: ",self.state)
    #print(active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name))

  def render(self):  #mode='human'
    return None
  def close (self):
    return None


# In[316]:


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
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


# In[317]:


env = CustomEnv()
env.BUDGET = 100
env = Monitor(env, log_dir)
check_env(env)


# In[318]:


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[64,64],
                                           layer_norm=True,
                                           feature_extraction="mlp")


# In[ ]:


log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
call_back = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

#model = DQN(CustomDQNPolicy, env, gamma=1,batch_size=128, verbose=1)
#policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256,256,128,128,64])
#model = PPO2("MlpPolicy",env,gamma = 0.9,verbose=1,nminibatches=32)
#model = DQN("MlpPolicy", env, gamma = 0.99,verbose=1)
model.learn(200000,callback = call_back)
model.save("dqn")


# In[327]:


results_plotter.plot_results([log_dir], 300000, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
plt.show()


# In[277]:



model = DQN(CustomDQNPolicy, env, gamma=1,verbose=1)
accuracies = 0
env.BUDGET = 100
for k in range(200):
    obs = env.reset()
    print(model)
    done = False
    i = 0
    average = 0
    while done == False:
        
        if random.randint(0,20) ==1 :
            
            det = True
        else:
            det = True
        action, _states = model.predict(obs, deterministic=det)
        print(model.action_probability(obs, state=None, mask=None, actions=None, logp=False))
        #print(MlpPolicy.step(obs,None, state=None, mask=None, deterministic=True))
        observation = obs.reshape((-1,) + model.observation_space.shape)
        print(model.step_model.step(observation)[1])
        #dif = model.step_model.step(observation)[1][0][1]-model.step_model.step(observation)[1][0][0]
        obs, rewards, done, info = env.step(action)
        print(action,rewards)
        env.render()
        i = i+1
    #env.print_status()
    accuracies = accuracies + active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name)
print(accuracies/200)


# In[214]:


plt.scatter(estimated_gains, real_gains)


# In[98]:


accuracies = 0 
for k in range(500):
    obs = env.reset()
    print(model)
    done = False
    i = 0
    while done == False:
        if i < 5:
            det = True
        else:
            det = True
        action = random.choice([0,1,2])
        #action, _states = model.predict(obs, deterministic=det)
        #print(model.action_probability(obs, state=None, mask=None, actions=None, logp=False))
        #print(MlpPolicy.step(obs,None, state=None, mask=None, deterministic=True))
        #observation = obs.reshape((-1,) + model.observation_space.shape)
        #print(model.step_model.step(observation))
        obs, rewards, done, info = env.step(action)
        print(action,rewards)
        env.render()
        i = i+1
    #env.print_status()
    accuracies = accuracies + active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),model_name)
print(accuracies/500)


# In[ ]:




