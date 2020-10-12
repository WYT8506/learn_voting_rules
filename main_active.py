from voting_rule import voting_rules
from voting_rule import learn_voting_rules
from voting_rule import events_likelyhood
from voting_rule import labeling
from machine_learning import KNN
from machine_learning import SVM
from fair import fairness
from criterias import criteria_satisfaction
from data import data_generator
from criterias import criteria_satisfaction
from sklearn.model_selection import train_test_split
import pickle
import contextlib
import io
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from active import active_learning
from active import generate_by_axioms

if __name__ == "__main__":

    candidates = ['a','b','c']
    preference_profiles = data_generator.generate_samples(candidates,['1']*10,500)
    preference_profiles_test = data_generator.generate_samples(candidates,['1']*10,500)

    params = {'objective':'multi:softprob', 'learning_rate' : 0.2,'max_depth' :10, 'n_estimators':30,'num_class':len(candidates)}
    #params = {'max_depth': 2, 'objective': 'multi:softprob','num_class':len(candidates)}
    Xs_test = labeling.get_Xs(candidates,preference_profiles_test)
    labels_test = labeling.get_labels(candidates,preference_profiles_test,voting_rules.voting_rule2A)
    total_Xs =[]
    total_ys = []
    for i in range(100):
        if i == 0:
            pfs = preference_profiles[0:1]
            labels = labeling.get_labels(candidates,pfs,voting_rules.voting_rule2A)
        else:
            pf = active_learning.choose_pf(candidates,preference_profiles,'model1')
            winner = candidates[labeling.get_labels(candidates,[pf],voting_rules.voting_rule2A)[0]]#pretend this is the winner labeled by the expert
            pfs,winners = active_learning.generate_by_axiom(candidates,pf,generate_by_axioms.generate_by_neutrality,0,winner)
            labels = labeling.winners_to_labels(candidates,winners)
            preference_profiles.remove(pf)
            print(pf)            
        Xs = labeling.get_Xs(candidates,pfs)
        total_Xs.extend(Xs)
        total_ys.extend(labels)
        print("added Xs: ",Xs)
        print("added Labels: ",labels)
        if i == 0:
            active_learning.initialize_model(np.array(Xs),np.array(labels),params,'model1')
        else:
            active_learning.initialize_model(np.array(total_Xs),np.array(total_ys),params,'model1')
            #active_learning.update_model(np.array(Xs),np.array(labels),params,'model1')
        active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),'model1')

