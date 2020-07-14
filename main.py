from voting_rule import voting_rules
from voting_rule import learn_voting_rules
from voting_rule import events_likelyhood
from machine_learning import KNN
from fair import fairness
from criterias import criteria_satisfaction
from data import data_generator
from criterias import criteria_satisfaction
from sklearn.model_selection import train_test_split
import pickle
if __name__ == "__main__":
    """
    d = data_generator(['a','b','c'])
    preference_profile = []
    for i in range(3):
       preference_profile.append(d.random_ranking())
    print(preference_profile)
    d.print_matrix(d.positional_score_matrix(preference_profile))
    d.print_matrix(d.weighted_majority_graph(preference_profile))
    winner = voting_rules.Borda(d.positional_score_matrix(preference_profile))
    print(winner)
    """
    tests = [10000]
    for sample_size in tests:
        print("sample size is:",str(sample_size))
        candidates = ['a','b','c']
        preference_profiles = data_generator.generate_samples(candidates,['1']*10,100000)
        learner = learn_voting_rules(candidates,preference_profiles)    
        preference_profiles_test = data_generator.generate_samples(candidates,['1']*10,sample_size)
        #criteria_satisfaction.tie_percentage(candidates,preference_profiles,voting_rules.Condorcet_Borda)
        #criteria_satisfaction.consistency_satisfaction(candidates,preference_profiles,2000,voting_rules.Condorcet_Borda)
        #criteria_satisfaction.consistency_satisfaction(candidates,preference_profiles,2000,voting_rules.voting_rule2)
        #criteria_satisfaction.consistency_satisfaction(candidates,preference_profiles,2000,voting_rules.voting_rule3)
        #criteria_satisfaction.condorcet_satisfaction(['a','b','c','d','e'],preference_profiles,voting_rules.Condorcet_Borda)
        """
        criteria_satisfaction.condorcet_satisfaction(['a','b','c','d','e'],preference_profiles,voting_rules.Condorcet_Borda)
        criteria_satisfaction.condorcet_satisfaction(['a','b','c','d','e'],preference_profiles,voting_rules.Condorcet)
        criteria_satisfaction.condorcet_satisfaction(['a','b','c','d','e'],preference_profiles,voting_rules.Copland)
        criteria_satisfaction.condorcet_satisfaction(['a','b','c','d','e'],preference_profiles,voting_rules.Minimax)

        criteria_satisfaction.consistency_satisfaction(['a','b','c','d','e'],preference_profiles,2000,voting_rules.Condorcet_Borda)
        criteria_satisfaction.consistency_satisfaction(['a','b','c','d','e'],preference_profiles,2000,voting_rules.Condorcet)
        criteria_satisfaction.consistency_satisfaction(['a','b','c','d','e'],preference_profiles,2000,voting_rules.Copland)
        criteria_satisfaction.consistency_satisfaction(['a','b','c','d','e'],preference_profiles,2000,voting_rules.Minimax)
        """
        #criteria_satisfaction.voting_rule_similarity(candidates,preference_profiles,voting_rules.voting_rule5,voting_rules.voting_rule1)
        #criteria_satisfaction.voting_rule_similarity(candidates,preference_profiles,voting_rules.voting_rule5,voting_rules.voting_rule2)
        #criteria_satisfaction.voting_rule_similarity(candidates,preference_profiles,voting_rules.voting_rule5,voting_rules.voting_rule3)
        #criteria_satisfaction.consistency_satisfaction(candidates,preference_profiles,2000,voting_rules.voting_rule5)
        #learner.learn_voting_rule(voting_rules.voting_rule4)

        """
        criteria_satisfaction.tie_percentage(candidates,preference_profiles,voting_rules.Borda)
        
        print(voting_rules.MPSR_tiebreaking(['a','b','c'],[['b','a','c']],['a','b']))
        
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,1000,voting_rules.voting_rule0)
        
        events_likelyhood.event_likelyhood(candidates,preference_profiles_test,10000,events_likelyhood.has_condorcet_winner)
        

        learner.learn_voting_rule(voting_rules.voting_rule1)
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,25000,None,loaded_model)
        criteria_satisfaction.neutrality_satisfaction(candidates,preference_profiles_test,25000,None,loaded_model)
        learner.learn_voting_rule(voting_rules.voting_rule2)
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,25000,None,loaded_model)
        criteria_satisfaction.neutrality_satisfaction(candidates,preference_profiles_test,25000,None,loaded_model)
        learner.learn_voting_rule(voting_rules.voting_rule3)
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,25000,None,loaded_model)
        criteria_satisfaction.neutrality_satisfaction(candidates,preference_profiles_test,25000,None,loaded_model)
        
        learner.learn_voting_rule(voting_rules.voting_rule3)
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
        """
        #criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,1000,None,loaded_model)
        

        #criteria_satisfaction.consistency_satisfaction(candidates,preference_profiles,2000,None,loaded_model)
        #criteria_satisfaction.condorcet_satisfaction(candidates,preference_profiles,None,loaded_model)
    #print(voting_rules.alpha_efficient_fair_borda(['a','b','c'],[['a','c','b'],['a','c','b'],['b','c','a']],[0,1],[2],0.5))
    #criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,10000,voting_rules.voting_rule10)
    #criteria_satisfaction.tie_percentage(candidates,preference_profiles_test,voting_rules.alpha_efficient_fair_borda)
    #criteria_satisfaction.unfairness(candidates,preference_profiles,10000,voting_rules.voting_rule10, [0,1,2,3,4],[5,6,7,8,9],learned_model = None)
    #criteria_satisfaction.unfairness(candidates,preference_profiles,10000,voting_rules.voting_rule1, [0,1,2,3,4],[5,6,7,8,9],learned_model = None)
    #criteria_satisfaction.unfairness(candidates,preference_profiles,10000,voting_rules.voting_rule2, [0,1,2,3,4],[5,6,7,8,9],learned_model = None)
    #criteria_satisfaction.unfairness(candidates,preference_profiles,10000,voting_rules.voting_rule3, [0,1,2,3,4],[5,6,7,8,9],learned_model = None)
    #criteria_satisfaction.consistency_satisfaction(candidates,preference_profiles_test,10000,voting_rules.voting_rule10)
    #learner.learn_voting_rule(voting_rules.voting_rule10)
    labels = []
    for pp in preference_profiles:
        labels.append(voting_rules.voting_rule10(candidates,pp))

    count =0
    for i in range(100):
        winner=KNN.get_winner(candidates,preference_profiles,labels,preference_profiles_test[i],voting_rules.voting_rule10)
        if winner == voting_rules.voting_rule10(candidates,preference_profiles_test[i]):
            count+=1
            print('correct')
        else:print('wrong')
        print(i/100,"%")
    print(count/100)


        
    """
    d = data_generator(['a','b','c']),
    pps,psms,wmgs = d.generate_samples(100,5)
    d.print_matrix(wmgs[0])
    borda_labels = []
    condorset_labels = []
    for psm in psms:
        borda_labels.append(voting_rules.Borda(psm))
    for wmg in wmgs:
        condorset_labels.append(voting_rules.Condorcet(wmg))
    print(borda_labels)
    print(condorset_labels)
    print(criteria_satisfaction.condorset_satisfaction(wmgs,borda_labels))
    """