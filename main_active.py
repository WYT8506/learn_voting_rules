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
import random
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
    preference_profiles = data_generator.generate_samples(candidates,['1']*50,20000)
    preference_profiles_test = data_generator.generate_samples(candidates,['1']*50,20000)

    params = {'objective':'multi:softprob', 'max_depth' :15,'n_estimators':30, 'num_class':len(candidates)}
    #params = {'objective':'multi:softprob', 'max_depth' :10, 'n_estimators':30,'num_class':len(candidates)}
    #params = {'max_depth': 2, 'objective': 'multi:softprob','num_class':len(candidates)}
    Xs_test = labeling.get_Xs(candidates,preference_profiles_test)
    labels_test = labeling.get_labels(candidates,preference_profiles_test,voting_rules.voting_rule2A)
    
    with open("pf.txt", "rb") as fp:   # Unpickling
        preference_profiles = pickle.load(fp)
    with open("test_pf.txt", "rb") as fp:   # Unpickling
        test_preference_profiles = pickle.load(fp)
    Xs_test = labeling.get_Xs(candidates,preference_profiles_test)
    labels_test = labeling.get_labels(candidates,preference_profiles_test,voting_rules.voting_rule2A)
    total_Xs =[]
    total_labels = []
    size = 1
    select_from_pool_size = 20
    num_data = []
    accuracies = []
    for i in range(500):
        if i == 0:
            new_pfs = preference_profiles[0:10]
            new_labels = labeling.get_labels(candidates,new_pfs,voting_rules.voting_rule2A)
        else:
            new_labels = []
            new_pfs = []
            select_from_pool = []
            for k in range(select_from_pool_size):
                select_from_pool.append(random.choice(preference_profiles))
            for j in range(size):
                pf = active_learning.choose_pf(candidates,select_from_pool,'model2',no_group = 1)
                winner = candidates[labeling.get_labels(candidates,[pf],voting_rules.voting_rule2A)[0]]#pretend this is the winner labeled by the expert
                #pfs,winners = active_learning.generate_by_axiom(candidates,pf,generate_by_axioms.generate_by_neutrality,0,winner)
                pfs = [pf]
                winners = [winner]
                labels = labeling.winners_to_labels(candidates,winners)
                preference_profiles.remove(pf)
                select_from_pool.remove(pf)
                new_pfs.extend(pfs)
                new_labels.extend(labels)
                #print(pf)           
        total_labels.extend(new_labels) 
        new_Xs = labeling.get_Xs(candidates,new_pfs)
        total_Xs.extend(new_Xs)
        print(len(total_Xs))
        print(len(total_labels))
        #print("added Xs: ",Xs)
        #print("added Labels: ",labels)
        if i == 0:
            active_learning.initialize_model(np.array(new_Xs),np.array(new_labels),params,'model2')
        else:
            active_learning.initialize_model(np.array(total_Xs),np.array(total_labels),params,'model2')
            #active_learning.update_model(np.array(new_Xs),np.array(new_labels),params,'model2')
        active_learning.prediction_accuracy(np.array(total_Xs),np.array(total_labels),'model2')
        accuracies.append(active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),'model2'))
        num_data.append(i*size)

    with open("n2.txt", "wb") as fp:   #Pickling
        pickle.dump(num_data, fp)
    with open("a2.txt", "wb") as fp:   #Pickling
        pickle.dump(accuracies, fp)
    

    """
    total_labels = []
    total_Xs = []
    size = 5
    for i in range(100):
        new_labels = []
        new_pfs = []
        for j in range(size):
            pf = random.choice(preference_profiles)
            winner = candidates[labeling.get_labels(candidates,[pf],voting_rules.voting_rule2A)[0]]#pretend this is the winner labeled by the expert
            pfs,winners = active_learning.generate_by_axiom(candidates,pf,generate_by_axioms.generate_by_neutrality,0,winner)
            labels = labeling.winners_to_labels(candidates,winners)
            new_pfs.extend(pfs)
            new_labels.extend(labels)
            preference_profiles.remove(pf)
        print(len(new_labels))
        total_labels.extend(new_labels)
        #print(len(new_pfs))

        #print(len(preference_profiles))
        new_Xs = labeling.get_Xs(candidates,new_pfs)
        total_Xs.extend(new_Xs)
        print(len(total_Xs))
        print(len(total_labels))
        #print("added Xs: ",Xs)
        #print("added Labels: ",labels)
        if i == 0:
            active_learning.initialize_model(np.array(new_Xs),np.array(new_labels),params,'model2')
        else:
            active_learning.initialize_model(np.array(total_Xs),np.array(total_labels),params,'model2')
            #active_learning.update_model(np.array(new_Xs),np.array(new_labels),params,'model2')
        active_learning.prediction_accuracy(np.array(total_Xs),np.array(total_labels),'model2')
        active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),'model2')

    """
    """
    num_data = []
    accuracies = []

    total_labels = []
    total_Xs = []
    size = 1
    for i in range(500):
        new_labels = []
        new_pfs = []
        for j in range(size):
            pf = random.choice(preference_profiles)
            winner = candidates[labeling.get_labels(candidates,[pf],voting_rules.voting_rule2A)[0]]#pretend this is the winner labeled by the expert
            pfs,winners = [pf],[winner]
            labels = labeling.winners_to_labels(candidates,winners)
            new_pfs.extend(pfs)
            new_labels.extend(labels)
            preference_profiles.remove(pf)
        print(len(new_labels))
        total_labels.extend(new_labels)
        #print(len(new_pfs))

        #print(len(preference_profiles))
        new_Xs = labeling.get_Xs(candidates,new_pfs)
        total_Xs.extend(new_Xs)
        print(len(total_Xs))
        print(len(total_labels))
        #print("added Xs: ",Xs)
        #print("added Labels: ",labels)
        if i == 0:
            active_learning.initialize_model(np.array(new_Xs),np.array(new_labels),params,'model1')
        else:
            active_learning.initialize_model(np.array(total_Xs),np.array(total_labels),params,'model1')
            #active_learning.update_model(np.array(new_Xs),np.array(new_labels),params,'model1')
        active_learning.prediction_accuracy(np.array(total_Xs),np.array(total_labels),'model1')
        accuracy = active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),'model1')
        num_data.append(i*size)
        accuracies.append(accuracy)
    with open("n1.txt", "wb") as fp:   #Pickling
        pickle.dump(num_data, fp)
    with open("a1.txt", "wb") as fp:   #Pickling
        pickle.dump(accuracies, fp)
    with open("pf.txt", "wb") as fp:   #Pickling
        pickle.dump(preference_profiles, fp)
    with open("test_pf.txt", "wb") as fp:   #Pickling
        pickle.dump(preference_profiles_test, fp)
    """