from voting_rules import voting_rules
from voting_rules import learn_voting_rules
from criterias import criteria_satisfaction
from data_generator import data_generator
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
    tests = [25000]
    for sample_size in tests:
        print("sample size is:",str(sample_size))
        candidates = ['a','b','c']
        preference_profiles = data_generator.generate_samples(candidates,['1']*100,sample_size)
        learner = learn_voting_rules(candidates,preference_profiles)    
        preference_profiles_test = data_generator.generate_samples(candidates,['1']*100,sample_size)
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
        criteria_satisfaction.tie_percentage(candidates,preference_profiles,voting_rules.Copland)
        criteria_satisfaction.tie_percentage(candidates,preference_profiles,voting_rules.Minimax)
        criteria_satisfaction.tie_percentage(candidates,preference_profiles,voting_rules.Condorcet_Borda)
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,25000,voting_rules.voting_rule1)
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,25000,voting_rules.voting_rule2)
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,25000,voting_rules.voting_rule3)
        """
        criteria_satisfaction.voting_rule_similarity(candidates,preference_profiles,voting_rules.voting_rule7,voting_rules.voting_rule1)
        criteria_satisfaction.voting_rule_similarity(candidates,preference_profiles,voting_rules.voting_rule7,voting_rules.voting_rule1)
        criteria_satisfaction.voting_rule_similarity(candidates,preference_profiles,voting_rules.voting_rule7,voting_rules.voting_rule2)
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,25000,voting_rules.voting_rule7)
        """
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
        """
        learner.learn_voting_rule(voting_rules.voting_rule7)
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
        criteria_satisfaction.neutrality_satisfaction1(candidates,preference_profiles_test,1000,None,loaded_model)
        criteria_satisfaction.neutrality_satisfaction(candidates,preference_profiles_test,1000,None,loaded_model)
        

        #criteria_satisfaction.consistency_satisfaction(candidates,preference_profiles,2000,None,loaded_model)
        #criteria_satisfaction.condorcet_satisfaction(candidates,preference_profiles,None,loaded_model)

    
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