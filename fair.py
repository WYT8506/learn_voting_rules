class fairness:
    def utility_function(alternative,ranking):
        return len(ranking)-ranking.index(alternative)-1

    def get_group_utility(candidates,preference_profile,alternative,group,utility_function):
        utility = 0
        for voter_index in group:
            utility+=utility_function(alternative,preference_profile[voter_index])
        utility=utility/len(group)
        return utility
    def get_social_welfare(candidates,preference_profile,alternative,utility_function):
        utility = 0
        for voter_index in range(len(preference_profile)):
            utility+=utility_function(alternative,preference_profile[voter_index])
        utility=utility/len(preference_profile)
        return utility

    def get_unfairness(candidates,preference_profile,alternative,group1,group2,utility_function):
        union = set(group1).union(set(group2))
        group_utility = fairness.get_group_utility(candidates,preference_profile,alternative,union,utility_function)
        if group_utility == 0:
            return 10000
        unfairness = abs((fairness.get_group_utility(candidates,preference_profile,alternative,group1,utility_function)\
            -fairness.get_group_utility(candidates,preference_profile,alternative,group2,utility_function)))\
        /fairness.get_group_utility(candidates,preference_profile,alternative,union,utility_function)
        return unfairness