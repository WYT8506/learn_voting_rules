
from machine_learning import learn
from data import data_generator
from fair import fairness
import random
class voting_rules:
    #correspondences
    def Borda(candidates,preference_profile):
        positional_score_matrix = data_generator.positional_score_matrix(candidates,preference_profile)
        n = len(positional_score_matrix)
        winner = []
        max_score = 0
        for candidate in positional_score_matrix:
            score = 0
            for i in range(n):
                s = n - i - 1
                score += candidate[i][1] * s

            if score > max_score:
                max_score = score
                winner = []
                winner.append(candidate[0][0][0])
            elif score == max_score:
                winner.append(candidate[0][0][0])
        return winner
    def Condorcet(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        for row in weighted_majority_graph:
            not_win = 0
            for column in row:
                if column[1]==0:
                    not_win +=1
            if not_win ==1:
                return column[0][0]
        return None
    def Condorcet_Borda(candidates,preference_profile):
        condorcet_winner = voting_rules.Condorcet(candidates,preference_profile)
        if condorcet_winner != None:
            return [condorcet_winner]
        else:
            borda_winner = voting_rules.Borda(candidates,preference_profile)
            return borda_winner
    def alpha_efficient_fair_borda(candidates,preference_profile,group1,group2,alpha):
        unfairness_list = []
        for candidate in candidates:
            unfairness = fairness.get_unfairness(candidates,preference_profile,candidate,group1,group2,fairness.utility_function)
            unfairness_list.append(unfairness)
        #print(unfairness_list)
        social_welfare_list = []
        for candidate in candidates:
            social_welfare = fairness.get_social_welfare(candidates,preference_profile,candidate,fairness.utility_function)
            social_welfare_list.append(social_welfare)
        #print(social_welfare_list)

        minimum = max(social_welfare_list)*alpha
        winners = []
        min_unfair = 10000
        for i in range(len(candidates)):
            if social_welfare_list[i]>=minimum:
                #print(">min",str(social_welfare_list[i]))
                if unfairness_list[i]<min_unfair:
                    winners = []
                    winners.append(candidates[i])
                    min_unfair = unfairness_list[i]
                elif abs(unfairness_list[i]-min_unfair) ==0:
                    winners.append(candidates[i])
        """
        if(len(winners)>1):
            print(social_welfare_list)
            print(unfairness_list)
        """
        return winners 





    def Copland(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winner = []
        max_wins = 0
        for i in range(len(weighted_majority_graph)):
            wins = 0
            for j in range(len(weighted_majority_graph[0])):
                if weighted_majority_graph[i][j][1]>weighted_majority_graph[j][i][1]:
                    wins +=1
                elif weighted_majority_graph[i][j][1]<weighted_majority_graph[j][i][1]:
                    wins -=1
            if wins > max_wins:
                max_wins= wins
                winner = []
                winner.append(weighted_majority_graph[i][j][0][0])
            elif wins == max_wins:
                winner.append(weighted_majority_graph[i][j][0][0])
        return winner
    def Minimax(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winner = []
        minimax_loss = 1000
        for i in range(len(weighted_majority_graph)):
            max_loss = 0
            for j in range(len(weighted_majority_graph[0])):
                loss = -weighted_majority_graph[i][j][1]+weighted_majority_graph[j][i][1]
                if loss > max_loss:
                    max_loss = loss
            if max_loss < minimax_loss:
                minimax_loss = max_loss
                winner = []
                winner.append(weighted_majority_graph[i][j][0][0])
            elif max_loss == minimax_loss:
                winner.append(weighted_majority_graph[i][j][0][0])
        return winner

    #tiebreakers
    def minimax_tiebreaking(candidates,preference_profile,winners):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winner = []
        minimax_loss = 1000
        for i in range(len(weighted_majority_graph)):
            max_loss = 0
            for j in range(len(weighted_majority_graph[0])):
                loss = -weighted_majority_graph[i][j][1]+weighted_majority_graph[j][i][1]
                if loss > max_loss:
                    max_loss = loss
            if max_loss < minimax_loss and (weighted_majority_graph[i][j][0][0] in winners):
                minimax_loss = max_loss
                winner = []
                winner.append(weighted_majority_graph[i][j][0][0])
            elif max_loss == minimax_loss and (weighted_majority_graph[i][j][0][0] in winners):
                winner.append(weighted_majority_graph[i][j][0][0])
        return winner
    def MPSR_tiebreaking(candidates,preference_profile,winners):
        max_rankings = []
        max = 0
        histogram = data_generator.histogram(candidates,preference_profile)
        #print(histogram)
        for e in histogram:
            if e[1]> max:
                max_rankings = []
                max_rankings.append(e[0])
                max = e[1]
            elif e[1] ==max:
                max_rankings.append(e[0])
        #print(max_rankings)
        if len(max_rankings)==1:
            max_ranking = max_rankings[0]
            for i in range(len(max_ranking)):
                if max_ranking[i] in winners:
                    return max_ranking[i]
        else:
            return None
    def voter1_tiebreaking(candidates,preference_profile,winners):
        ranking = preference_profile[0]
        for candidate in ranking:
            if candidate in winners:
                return candidate

    #voting rules
    def voting_rule0(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winners = voting_rules.Borda(candidates,preference_profile)
        #return random.choice(winners)
        #return max(voting_rules.minimax_tiebreaking(candidates,winners,preference_profile))
        mpsr = voting_rules.MPSR_tiebreaking(candidates,preference_profile,winners)
        
        if mpsr !=None:
            return mpsr
        
        return max(winners)
    def voting_rule1(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winners = voting_rules.Borda(candidates,preference_profile)
        #return random.choice(winners)
        #return max(voting_rules.minimax_tiebreaking(candidates,winners,preference_profile))
        """
        mpsr = voting_rules.MPSR_tiebreaking(candidates,preference_profile,winners)
        
        if mpsr !=None:
            return mpsr
        """
        return max(winners)
    def voting_rule2A(candidates,preference_profile):
        winners = voting_rules.Copland(candidates,preference_profile)
   
        mpsr = voting_rules.MPSR_tiebreaking(candidates,preference_profile,winners)
        
        if mpsr !=None:
            return mpsr
   

        #return random.choice(winners)
        #return voting_rules.voter1_tiebreaking(candidates,preference_profile,winners)
        return max(winners)
    def voting_rule2B(candidates,preference_profile):
        winners = voting_rules.Copland(candidates,preference_profile)
        return max(winners)
    def voting_rule2C(candidates,preference_profile):
        winners = voting_rules.Copland(candidates,preference_profile)
        return voting_rules.voter1_tiebreaking(candidates,preference_profile,winners)

    def voting_rule2D(candidates,preference_profile):
        winners = voting_rules.Copland(candidates,preference_profile)
   
   

        return random.choice(winners)

    def voting_rule10(candidates,preference_profile,group1 = None,group2 = None,alpha = None):
        if group1 == None:
            group1 = [0]
        if group2 == None:
            group2 = [1,2]
        if alpha == None:
            alpha = 0.5
        winners = voting_rules.alpha_efficient_fair_borda(candidates,preference_profile,group1,group2,alpha)
        mpsr = voting_rules.MPSR_tiebreaking(candidates,preference_profile,winners)
        
        if mpsr !=None:
            return mpsr
        #print(winners)
        return max(winners)

    def voting_rule3(candidates,preference_profile):
        winners = voting_rules.Minimax(candidates,preference_profile)
        return max(winners)
    def voting_rule4(candidates,preference_profile):
        winners = voting_rules.Condorcet_Borda(candidates,preference_profile)
        return max(winners)

    def voting_rule5(candidates,preference_profile):
        copland_winners = voting_rules.Copland(candidates,preference_profile)
        borda_winners = voting_rules.Borda(candidates,preference_profile)
        for winner in copland_winners:
            if winner in borda_winners:
                return winner
            else:
                return max(copland_winners)

    def voting_rule6(candidates,preference_profile):
        if voting_rules.voting_rule1(candidates,preference_profile) == voting_rules.voting_rule2(candidates,preference_profile):
            return voting_rules.voting_rule1(candidates,preference_profile)
                
        rule = random.choice(['Borda','Copland'])
        if rule == 'Borda':
           return voting_rules.voting_rule1(candidates,preference_profile)
        else:
            return voting_rules.voting_rule2(candidates,preference_profile)

    def voting_rule7(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winners = voting_rules.Borda(candidates,preference_profile)
        #return random.choice(winners)
        return max(voting_rules.minimax_tiebreaking(candidates,winners,preference_profile))

    def learned_rule(candidates,preference_profile,learned_model):
        positional_score_matrix = data_generator.positional_score_matrix(
        candidates,preference_profile)
        weighted_majority_graph = data_generator.weighted_majority_graph(
        candidates,preference_profile)
        features = data_generator.get_feature_vector(
                positional_score_matrix, weighted_majority_graph)
        x = learn.load_x(None,[features])
        return candidates[learned_model.predict(x)[0]]
    
class events_likelyhood:
    def event_likelyhood(candidates,preference_profiles,n,event):
        profile_count = 0
        count = 0
        for pp in preference_profiles:
            if events_likelyhood.has_condorcet_winner(candidates,pp)==True:
                count+=1
            profile_count+=1
            if(profile_count == n):
                break
        print(count/n)
        return (n-count)/n

    def has_condorcet_winner(candidates,preference_profile):
        if voting_rules.Condorcet(candidates,preference_profile)!=None:
            return True
        else: return False


class labeling:
    def get_Xs(candidates,preference_profiles):
        Xs = []
        for i in range(len(preference_profiles)):
            preference_profile = preference_profiles[i]
            positional_score_matrix = data_generator.positional_score_matrix(
                candidates,preference_profile)
            weighted_majority_graph = data_generator.weighted_majority_graph(
                candidates,preference_profile)
            histogram = data_generator.histogram(candidates,preference_profile)
            X = data_generator.get_feature_vector(
                positional_score_matrix, weighted_majority_graph, histogram,preference_profile,candidates)
            Xs.append(X)
        return Xs

    def get_labels(candidates,preference_profiles,voting_rule): 
        labels = []
        for pp in preference_profiles:
             labels.append(voting_rule(candidates,pp))
        ys = []
        for label in labels:
           ys.append(candidates.index(label)) 
        return ys
    def winners_to_labels(candidates,winners):
        labels = []
        for e in winners:
            labels.append(candidates.index(e))
        return labels

class learn_voting_rules:

    def __init__(self, candidates, preference_profiles):
        self.candidates = candidates
        self.preference_profiles = preference_profiles
    def get_preference_profiles(self):
        return self.preference_profiles
    def learn_voting_rule(self,voting_rule,one_vs_one = None):
        headers = []
        sample_set = []
        labels = []
        for i in range(len(self.preference_profiles)):
            preference_profile = self.preference_profiles[i]
            positional_score_matrix = data_generator.positional_score_matrix(
                self.candidates,preference_profile)
            weighted_majority_graph = data_generator.weighted_majority_graph(
                self.candidates,preference_profile)
            histogram = data_generator.histogram(self.candidates,preference_profile)
            x = data_generator.get_feature_vector(
                positional_score_matrix, weighted_majority_graph, histogram,preference_profile,self.candidates)
            y = voting_rule(
                self.candidates,preference_profile)
            feature_names = data_generator.get_feature_names(
                positional_score_matrix, weighted_majority_graph)
            if i == 0:
                headers = feature_names
            sample_set.append(x)
            labels.append(y)
        print(len(labels))
        x_data = learn.load_x(headers, sample_set)
        y_data = learn.load_y(self.candidates, labels)
        #print(x_data)
        #print(y_data)
        if (one_vs_one): learn.one_vs_one(x_data,y_data)
        else:learn.train(x_data, y_data)

 