from data_generator import data_generator
from voting_rules import voting_rules
from itertools import permutations
from learn import learn

import copy
import random
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
class criteria_satisfaction:
    def condorcet_satisfaction(candidates,perference_profiles,voting_rule,learned_model = None):
        count = 0
        voting_rule_count = 0
        count_total = 0
        for i in range(len(perference_profiles)):
            condorcet_winner = voting_rules.Condorcet(candidates,perference_profiles[i])
            if learned_model == None:
                winner = voting_rule(candidates,perference_profiles[i])
            else :
                winner = voting_rules.learned_rule(candidates,perference_profiles[i],learned_model)
                winner = candidates[winner]
            if winner != None:
                voting_rule_count+=1
            if condorcet_winner !=None:
                count_total +=1
            else:
                continue

            if winner == condorcet_winner:
                count +=1
        print(voting_rule_count)
        print(count/count_total)
        return count/count_total
    def voting_rule_similarity(candidates,perference_profiles,voting_rule1,voting_rule2):
        count = 0
        voting_rule_count = 0
        count_total = 0
        for i in range(len(perference_profiles)):
            borda_winner = voting_rule1(candidates,perference_profiles[i])
            winner = voting_rule2(candidates,perference_profiles[i])
            if len(winner) ==1:
                voting_rule_count+=1

            count_total +=1
                

            if max(winner) == max(borda_winner):
                count +=1
        print(count/count_total)
        return count/count_total
    def neutrality_satisfaction(candidates,preference_profiles,n,voting_rule,learned_model = None):
        profile_count = 0
        count = 0
        for pp in preference_profiles:
            """
            print(profile_count)
            print(str(pp[0]))
            """
            if learned_model == None:
                pp_win = voting_rule(candidates,pp)
            else:
                pp_win = voting_rules.learned_rule(candidates,pp,learned_model)
            for i in range(len(candidates)):
                new_pp = copy.deepcopy(pp)
                if candidates[i] == pp_win:
                    continue
                candidate1 = pp_win
                candidate2 = candidates[i]
                for j in range(len(new_pp)):
                    index1 = new_pp[j].index(candidate1)
                    index2 = new_pp[j].index(candidate2)
                    new_pp[j][index1], new_pp[j][index2] = new_pp[j][index2], new_pp[j][index1]

                if learned_model == None:
                    new_pp_win = voting_rule(candidates,new_pp)
                else:
                    new_pp_win = voting_rules.learned_rule(candidates,new_pp,learned_model)
                """
                print(profile_count)
                print(str(new_pp[0]))
                print(str(pp[0]))                    
                print(pp_win)
                print(new_pp_win)
                print(candidate2)
                """
                if new_pp_win != candidate2:
                    count+=1
                    break
                else:
                    continue
            #print(profile_count)
        print((n-count)/n)
        return (n-count)/n
    def neutrality_satisfaction1(candidates,preference_profiles,n,voting_rule,learned_model = None):
        profile_count = 0
        count = 0
        for pp in preference_profiles:
            """
            print(profile_count)
            print(str(pp[0]))
            """
            if learned_model == None:
                pp_win = voting_rule(candidates,pp)
            else:
                pp_win = voting_rules.learned_rule(candidates,pp,learned_model)
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

                if learned_model == None:
                    new_pp_win = voting_rule(candidates,new_pp)
                else:
                    new_pp_win = voting_rules.learned_rule(candidates,new_pp,learned_model)
                """
                print(profile_count)
                print(str(new_pp[0]))
                print(str(pp[0]))                    
                print(pp_win)
                print(new_pp_win)
                print(perm[candidates.index(pp_win)])
                """
                if new_pp_win != perm[candidates.index(pp_win)]:
                    count+=1
                    break
                else:
                    continue
            profile_count+=1
            if(profile_count == n):
                break
        print((n-count)/n)
        return (n-count)/n


    def consistency_satisfaction(candidates,preference_profiles,n,voting_rule,learned_model = None):
        profile_count = 0
        consistency_count = 0
        """
        clf = XGBClassifier()
        clf._Booster.load_model('..')
        """
        new_pps = []
        while profile_count<n:
            pp1 = random.choice(preference_profiles)
            pp2 = random.choice(preference_profiles)
            new_pp = pp1+pp2
            if learned_model == None:
                if(voting_rule(candidates,pp1)==voting_rule(candidates,pp2) and voting_rule(candidates,pp2)!=None):
                    profile_count+=1
                    if voting_rule(candidates,pp1) == voting_rule(candidates,new_pp):
                       consistency_count+=1
            else:

                if(voting_rules.learned_rule(candidates,pp1,learned_model)==voting_rules.learned_rule(candidates,pp2,learned_model)):
                    profile_count+=1
                    if voting_rules.learned_rule(candidates,pp1,learned_model)==voting_rules.learned_rule(candidates,new_pp,learned_model):
                       consistency_count+=1

        print(consistency_count/profile_count)

    def tie_percentage(candidates,perference_profiles,voting_rule):
        tie_count = 0
        count_total = 0
        for i in range(len(perference_profiles)):
            winner = voting_rule(candidates,perference_profiles[i])
            if len(winner)>1:
                tie_count+=1
                count_total +=1
            else:
                count_total+=1

        print(tie_count/count_total)
        return tie_count/count_total