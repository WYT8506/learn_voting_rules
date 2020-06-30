
from learn import learn
from data_generator import data_generator
import random

class voting_rules:
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
    def minimax_tiebreaking(candidates,winners, preference_profile):
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
    def voting_rule1(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winners = voting_rules.Borda(candidates,preference_profile)
        #return random.choice(winners)
        #return max(voting_rules.minimax_tiebreaking(candidates,winners,preference_profile))
        return max(winners)
    def voting_rule2(candidates,preference_profile):
        winners = voting_rules.Copland(candidates,preference_profile)
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
                positional_score_matrix, weighted_majority_graph, histogram)
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

 