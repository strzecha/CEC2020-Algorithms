"""Module with basic mutation strategies from Differential Evolution
"""

def current_to_pbest_with_archive(x, P, P_with_archive, r, pbest):
        return x + x.F * (pbest - x + P[r[0]] - P_with_archive[r[2]])

def current_to_pbest_without_archive(x, P, P_with_archive, r, pbest):
    return x + x.F * (pbest - x + P[r[0]] - P[r[1]])

def weighted_rand_to_pbest(x, P, P_with_archive, r, pbest):
        return x.F * P[r[0]] + x.F * (pbest - P[r[1]])
