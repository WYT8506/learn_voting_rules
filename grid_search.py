
import os
if os.path.isfile("model3"):
   os.remove("model3")
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

if __name__ == "__main__":
    voting_rule = voting_rules.voting_rule2A
    candidates = [0,1,2,3]
    preference_profiles = data_generator.generate_samples(candidates,['1']*50,10000)
    preference_profiles_test = data_generator.generate_samples(candidates,['1']*50,50000)
    combinations = []
    max_depth = [5,10,15,20]
    max_number = [10,20,30,40]
    for depth in max_depth:
        for number in max_number:
            combinations.append((depth,number))
    training_accuracies = []
    testing_accuracies = []
    for combination in combinations:
        params = {'objective':'multi:softprob', 'max_depth' :combination[0],'n_estimators':combination[1], 'num_class':len(candidates)}
        Xs_train = labeling.get_Xs(candidates,preference_profiles)
        labels_train = labeling.get_labels(candidates,preference_profiles,voting_rule)
        Xs_test = labeling.get_Xs(candidates,preference_profiles_test)
        labels_test = labeling.get_labels(candidates,preference_profiles_test,voting_rule)
        active_learning.initialize_model(np.array(Xs_train),np.array(labels_train),params,'model3')
        training_accuracies.append(active_learning.prediction_accuracy(np.array(Xs_train),np.array(labels_train),'model3'))
        testing_accuracies.append(active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),'model3'))
    print(combinations)
    print(training_accuracies)
    print(testing_accuracies)