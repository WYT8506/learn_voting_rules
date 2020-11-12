
from data_new import data_generator
import xgboost as xgb
import random
import numpy as np
class voting_rules:
    #correspondences
    def Borda(candidates,preference_profile):
        positional_score_matrix = data_generator.positional_score_matrix(candidates,preference_profile)
        n = len(positional_score_matrix)
        winner = []
        max_score = 0
        for candidate_index in range(n):
            candidate = positional_score_matrix[candidate_index]
            score = 0
            for i in range(n):
                s = n - i - 1
                score += candidate[i] * s

            if score > max_score:
                max_score = score
                winner = []
                winner.append(candidate_index)
            elif score == max_score:
                winner.append(candidate_index)
        return winner
    def plurality_winner(preference_profiles):
        votes = np.array(preference_profiles)
        n, m = votes.shape
        scores = np.zeros(m)
        for i in range(n):
            scores[votes[i][0]] += 1
        winner = np.argwhere(scores == np.max(scores)).flatten().tolist()
    
        return winner, scores
    
    def STV_winner(candidates, preference_profiles):
        votes_cpy = preference_profiles.copy()
        votes_cpy = np.array(votes_cpy)
        n, m = votes_cpy.shape
        return voting_rules.STV_helper(votes_cpy, n, m, [])[0]

    def STV_helper(votes, n, m, removed):
        """
        Parameters
        ----------
        votes : preference profile
        n : #votes
        m : #candidates in original election
        removed : already removed candidates
        """
        winner, scores = voting_rules.plurality_winner(votes)
        
        if(np.max(scores) >= n/2):
            return winner, scores
        rest_scores = scores
        rest_scores[removed] = np.inf
        c_last = np.argmin(rest_scores)
        removed.append(c_last)
        new_votes = []
        for v in votes:
            newv = np.delete(v, np.where(v==c_last))
            newv = np.append(newv, c_last)
            
            new_votes.append(newv)
        
        return STV_helper(np.array(new_votes), n, m, removed)

    def Copland(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winner = []
        max_wins = 0
        for i in range(len(weighted_majority_graph)):
            wins = 0
            for j in range(len(weighted_majority_graph[0])):
                if weighted_majority_graph[i][j]>weighted_majority_graph[j][i]:
                    wins +=1
                elif weighted_majority_graph[i][j]<weighted_majority_graph[j][i]:
                    wins -=1
            if wins > max_wins:
                max_wins= wins
                winner = []
                winner.append(i)
            elif wins == max_wins:
                winner.append(i)
        return winner
    def Minimax(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winner = []
        minimax_loss = 1000
        for i in range(len(weighted_majority_graph)):
            max_loss = 0
            for j in range(len(weighted_majority_graph[0])):
                loss = -weighted_majority_graph[i][j]+weighted_majority_graph[j][i]
                if loss > max_loss:
                    max_loss = loss
            if max_loss < minimax_loss:
                minimax_loss = max_loss
                winner = []
                winner.append(i)
            elif max_loss == minimax_loss:
                winner.append(i)
        return winner

    #tiebreakers
    def minimax_tiebreaking(candidates,preference_profile,winners):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winner = []
        minimax_loss = 1000
        for i in range(len(weighted_majority_graph)):
            max_loss = 0
            for j in range(len(weighted_majority_graph[0])):
                loss = -weighted_majority_graph[i][j]+weighted_majority_graph[j][i]
                if loss > max_loss:
                    max_loss = loss
            if max_loss < minimax_loss and (i in winners):
                minimax_loss = max_loss
                winner = []
                winner.append(i)
            elif max_loss == minimax_loss and (i in winners):
                winner.append(i)
        return winner
    def MPSR_tiebreaking(candidates,preference_profile,winners):
        max_rankings = []
        Max = 0
        histogram = data_generator.histogram(candidates,preference_profile)
        #print(histogram)
        for e in histogram:
            if e[1]> Max:
                max_rankings = []
                max_rankings.append(e[0])
                Max = e[1]
            elif e[1] ==Max:
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

    def voting_rule7(candidates,preference_profile):
        weighted_majority_graph = data_generator.weighted_majority_graph(candidates,preference_profile)
        winners = voting_rules.Borda(candidates,preference_profile)
        #return random.choice(winners)
        return max(voting_rules.minimax_tiebreaking(candidates,winners,preference_profile))

    def learned_rule(candidates,preference_profile,clf):
        positional_score_matrix = data_generator.positional_score_matrix(
        candidates,preference_profile)
        weighted_majority_graph = data_generator.weighted_majority_graph(
        candidates,preference_profile)
        histogram = data_generator.histogram(candidates,preference_profile)
        X = data_generator.get_feature_vector(
                positional_score_matrix, weighted_majority_graph, histogram,preference_profile,candidates)
        preds = clf.predict(xgb.DMatrix([X]))
        preds = np.array(preds)
        preds = np.argmax(preds,axis = 1)
        return candidates[preds[0]]
 


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
        return labels
    def winners_to_labels(candidates,winners):
        return winners
