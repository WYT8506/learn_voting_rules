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
                l.append([(candidates[i], j), 0])
            matrix.append(l)

        for i in range(len(preference_profile)):
            ranking = preference_profile[i]
            for j in range(len(ranking)):
                candidate_index = candidates.index(ranking[j])
                matrix[candidate_index][j][1]+=1
        """
        for i in range(num_candidates):
                for j in range(num_candidates):
                    matrix[i][j][1]= matrix[i][j][1]/len(preference_profile)
        """
        return matrix

    def weighted_majority_graph(candidates, preference_profile):
        num_candidates = len(preference_profile[0])
        graph_matrix = []
        for i in range(num_candidates):
            l = []
            for j in range(num_candidates):
                l.append([(candidates[i],candidates[j]),0])
            graph_matrix.append(l)

        for ranking in preference_profile:
            for i in range(num_candidates):
                for j in range(num_candidates):
                    if ranking.index(candidates[i])<ranking.index(candidates[j]):
                        graph_matrix[i][j][1]+=1
        
        for i in range(num_candidates):
            for j in range(num_candidates):
                if(graph_matrix[i][j][1]>graph_matrix[j][i][1]):
                    graph_matrix[i][j][1] = graph_matrix[i][j][1]-graph_matrix[j][i][1]
                    graph_matrix[j][i][1] = 0
                else:
                    graph_matrix[j][i][1] = graph_matrix[j][i][1]-graph_matrix[i][j][1]
                    graph_matrix[i][j][1] = 0
        """
        for i in range(num_candidates):
                for j in range(num_candidates):
                    graph_matrix[i][j][1]= graph_matrix[i][j][1]/len(preference_profile)
        """
        return graph_matrix
    def histogram(candidates,preference_profile):
        perm = list(permutations(candidates))
        perm = [''.join(ele) for ele in perm] 
        histogram = dict()
        return_list = []
        for permutation in perm:
            return_list.append([permutation, 0])
        rankings = [''.join(ele) for ele in preference_profile] 
        for permutation in perm:
            histogram[permutation] = 0
        for ranking in rankings:
            histogram[ranking]+=1
        for permutation in return_list:
            permutation[1] = histogram[permutation[0]]
        return return_list

    def get_feature_vector(positional_score_matrix, weighted_majority_graph, histogram=None):
        num_candidates = len(positional_score_matrix)
        vector = []
        """
        for i in range(num_candidates):
            for j in range(num_candidates):
                vector.append(positional_score_matrix[i][j][1])
        
        for i in range(num_candidates):
            for j in range(num_candidates):
                vector.append(weighted_majority_graph[i][j][1])
        """
        for permutation in histogram:
            vector.append(permutation[1])
        #print(vector)
        
        return vector

    def get_feature_names(positional_score_matrix, weighted_majority_graph):
        """
        self.print_matrix(positional_score_matrix)
        self.print_matrix(weighted_majority_graph)
        """
        num_candidates = len(positional_score_matrix)
        vector = []

        for i in range(num_candidates):
            for j in range(num_candidates):
                vector.append(positional_score_matrix[i][j][0])

        for i in range(num_candidates):
            for j in range(num_candidates):
                vector.append(weighted_majority_graph[i][j][0])
        return vector

