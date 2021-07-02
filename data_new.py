import random
from itertools import permutations
import collections

class data_generator:

    def random_ranking(candidates):
        num_candidates = len(candidates)
        random_ranking = random.sample(candidates,num_candidates)
        return random_ranking

    def generate_samples(candidates,voters,num_sample):
        preference_profiles = []

        for i in range(num_sample):
            preference_profile = []
            for j in range(len(voters)):
                preference_profile.append(data_generator.random_ranking(candidates))
          
            preference_profiles.append(preference_profile)
        return preference_profiles

    def print_matrix(matrix):
        for i in range(len(matrix)):
            s = ""
            for j in range(len(matrix[i])):
               s+=str(matrix[i][j])
            print(s)
    def positional_score_matrix(candidates, preference_profile):
        num_candidates = len(preference_profile[0])
        matrix = []
        for i in range(num_candidates):
            l = []
            for j in range(num_candidates):
                l.append(0)
            matrix.append(l)

        for i in range(len(preference_profile)):
            ranking = preference_profile[i]
            for j in range(len(ranking)):
                candidate = ranking[j]
                matrix[candidate][j]+=1
        return matrix

    def weighted_majority_graph(candidates, preference_profile):
        num_candidates = len(preference_profile[0])
        graph_matrix = []
        for i in range(num_candidates):
            l = []
            for j in range(num_candidates):
                l.append(0)
            graph_matrix.append(l)

        for ranking in preference_profile:
            for i in range(num_candidates):
                for j in range(i,num_candidates):
                    graph_matrix[ranking[i]][ranking[j]]+=1
        
        for i in range(num_candidates):
            for j in range(i,num_candidates):
                if(graph_matrix[i][j]>graph_matrix[j][i]):
                    graph_matrix[i][j] = graph_matrix[i][j]-graph_matrix[j][i]
                    graph_matrix[j][i] = 0
                else:
                    graph_matrix[j][i] = graph_matrix[j][i]-graph_matrix[i][j]
                    graph_matrix[i][j] = 0

        return graph_matrix

    def histogram(candidates,preference_profile):
        perm = list(permutations(candidates))
        histogram = dict()
        return_list = []
        for permutation in perm:
            return_list.append([permutation, 0])
        rankings = preference_profile
        for permutation in perm:
            histogram[tuple(permutation)] = 0
        for ranking in rankings:
            histogram[tuple(ranking)]+=1
        for permutation in return_list:
            permutation[1] = histogram[permutation[0]]
        return return_list

    def get_feature_vector(positional_score_matrix, weighted_majority_graph, histogram=None, preference_profile = None,candidates = None):
        num_candidates = len(positional_score_matrix)
        vector = []
        """
        perm = list(permutations(candidates))
        perm = [''.join(ele) for ele in perm] 
        for ranking in preference_profile:
            ranking = ''.join(ranking)
            vector.append(perm.index(ranking))
        """
        """
        for i in range(num_candidates):
            for j in range(num_candidates):
                vector.append(positional_score_matrix[i][j][1])
        
        
        for i in range(num_candidates):
            for j in range(num_candidates):
                vector.append(weighted_majority_graph[i][j][1])
        #print(preference_profile[0])
        for c in preference_profile[0]:
            vector.append(ord(c)-97)
        """
        for i in range(num_candidates):
            for j in range(num_candidates):
                vector.append(positional_score_matrix[i][j])
        for i in range(num_candidates):
            for j in range(num_candidates):
                vector.append(weighted_majority_graph[i][j])
        
        for permutation in histogram:
            vector.append(permutation[1])
        """
        for permutation in histogram:
            vector.append(permutation[1])
        """
        """
        #print(vector)
        for ranking in preference_profile:
            for i in range(len(candidates)):
                for j in range(i+1,len(candidates)):
                    if ranking.index(candidates[i])>ranking.index(candidates[j]):
                        vector.append(1)
                    else:
                        vector.append(0)
        """
        #print(vector)
        return vector

