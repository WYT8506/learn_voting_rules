
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

def consistency_satisfaction_model(candidates,preference_profiles,n,model_name):
    clf = xgb.Booster({'nthread': 4})  # init model
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
        preds_pp1=np.argmax(clf.predict(xgb.DMatrix(pp1_Xs)),axis = 1)
        preds_pp2=np.argmax(clf.predict(xgb.DMatrix(pp2_Xs)),axis = 1)
        preds_newpp=np.argmax(clf.predict(xgb.DMatrix(new_pp_Xs)),axis = 1)
        if preds_pp1==preds_pp2:
            profile_count+=1
            if preds_pp1==preds_newpp:
               consistency_count+=1
    return consistency_count/profile_count


if __name__ == "__main__":
    total = len(sys.argv)
 
    # Get the arguments list 
    cmdargs = str(sys.argv)
    trials=int(sys.argv[1])
    iteration = int(sys.argv[2])
    label_per_iteration = int(sys.argv[3])
    generate_per_label = int(sys.argv[4])
    rule = str(sys.argv[5])
    n_candidates = int(sys.argv[6])
    n_voters = int(sys.argv[7])
    satisfaction_samples = int(sys.argv[8])


    if rule == "borda":
        voting_rule = voting_rules.voting_rule0
    elif rule == "copland":
        voting_rule  = voting_rules.voting_rule2A


    candidates = list(range(0,n_candidates))
    print(candidates)
    #voters = ['1']*n_voters
    #candidates = [0,1,2]
    voters = ['1']*n_voters
    params = {'objective':'multi:softprob', 'max_depth' :15,'n_estimators':30, 'num_class':len(candidates)}
    #params = {'objective':'multi:softprob', 'max_depth' :10, 'n_estimators':30,'num_class':len(candidates)}
    #params = {'max_depth': 2, 'objective': 'multi:softprob','num_class':len(candidates)}

    preference_profiles = data_generator.generate_samples(candidates,voters,50000)
    preference_profiles_test = data_generator.generate_samples(candidates,voters,50000)
    with open("consistency_pf.txt", "wb") as fp:   #Pickling
        pickle.dump(preference_profiles, fp)
    with open("consistency_pf_test.txt", "wb") as fp:   #Pickling
        pickle.dump(preference_profiles_test, fp)

    """
    with open("consistency_pf.txt", "rb") as fp:   # Unpickling
        preference_profiles = pickle.load(fp)
    with open("consistency_pf_test.txt", "rb") as fp:   # Unpickling
        preference_profiles_test = pickle.load(fp)
    """
    Xs_test = labeling.get_Xs(candidates,preference_profiles_test)
    labels_test = labeling.get_labels(candidates,preference_profiles_test,voting_rule)
    average_accuracies_all = []
    average_consistency_satisfactions_all = []
    num_data = []
    combinations = [(True,True),(True,False),(False,True),(False,False)]
    for combination in combinations:
        print("Active,Aug: ",combination, "=======================" )
        active = combination[0]
        data_augmentation = combination[1]

        average_accuracies = []
        average_consistency_satisfactions = []

     
        for trial_number in range(trials):
            print("trial number is: ", combination,trial_number,"============================")
            total_pfs = []
            total_Xs =[]
            total_labels = []

            select_from_pool_size = 20
            num_data = []
            accuracies = []
            consistency_satisfactions = []

            for i in range(iteration):
                if i == 0:
                    new_pfs = preference_profiles[0:100]
                    new_labels = labeling.get_labels(candidates,new_pfs,voting_rule)
                else:
                    new_labels = []
                    new_pfs = []
                    select_from_pool = []
                    for k in range(select_from_pool_size):
                        select_from_pool.append(random.choice(preference_profiles))
                    for j in range(label_per_iteration):
                        if active == False: 
                            pf = random.choice(select_from_pool)
                        else:
                            pf = active_learning.choose_pf(candidates,select_from_pool,'model3',no_group = 1)
                        label = labeling.get_labels(candidates,[pf],voting_rule)[0]#pretend this is the winner labeled by the expert

                        if data_augmentation == True:
                            pfs,labels = generate_by_axioms.generate_by_consistency1(candidates,total_pfs,total_labels,pf,label,generate_per_label)
                        #pfs = [pf]
                        #winners = [winner]

                        preference_profiles.remove(pf)
                        select_from_pool.remove(pf)
                        new_pfs.append(pf)
                        new_labels.append(label)
                        if data_augmentation == True:
                            new_pfs.extend(pfs)
                            new_labels.extend(labels)

                        #print(pf)           
                total_labels.extend(new_labels) 
                #print(new_pfs[0])
                new_Xs = labeling.get_Xs(candidates,new_pfs)
                total_Xs.extend(new_Xs)
                total_pfs.extend(new_pfs)

                #print("added Xs: ",Xs)
                #print("added Labels: ",labels)
                if i == 0:
                    active_learning.initialize_model(np.array(new_Xs),np.array(new_labels),params,'model3')
                else:
                    active_learning.initialize_model(np.array(total_Xs),np.array(total_labels),params,'model3')
                    #active_learning.update_model(np.array(new_Xs),np.array(new_labels),params,'model2')
                if (i%(iteration/10) ==0):           
                    print(len(total_Xs))
                    print(len(total_labels))
                    active_learning.prediction_accuracy(np.array(total_Xs),np.array(total_labels),'model3')
                    accuracies.append(active_learning.prediction_accuracy(np.array(Xs_test),np.array(labels_test),'model3'))
                    cs=consistency_satisfaction_model(candidates,preference_profiles_test,satisfaction_samples,model_name = 'model3')
                    print("consistency_satisfaction:",cs)
                    consistency_satisfactions.append(cs)
                    num_data.append(i*label_per_iteration)
            average_accuracies.append(accuracies)
            average_consistency_satisfactions.append(consistency_satisfactions)
    
        print(average_accuracies)
        print(average_consistency_satisfactions)
        average_accuracies_all.append(average_accuracies)
        average_consistency_satisfactions_all.append(average_consistency_satisfactions)
    print(average_accuracies_all)
    print(average_consistency_satisfactions_all)
    print(num_data)
    with open("num_data.txt", "wb") as fp:   #Pickling
        pickle.dump(num_data, fp)
    with open("test_accuracy.txt", "wb") as fp:   #Pickling
        pickle.dump(average_accuracies_all, fp)
    with open("consistency_satisfaction.txt", "wb") as fp:   #Pickling
        pickle.dump(average_consistency_satisfactions_all, fp)
    