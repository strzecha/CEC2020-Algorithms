import numpy as np
import copy
import matplotlib.pyplot as plt

k_r = 0.5
k_f = 0.5
k = 0.7
p = 0.5

class AGSKIndividual:
    def __init__(self, x):
        self.x = x
        self.objective = 0

    def __repr__(self):
        return f"{self.x}: obj:{self.objective}"

    def __add__(self, individual):
        return AGSKIndividual(self.x + individual.x)
    
    def __sub__(self, individual):
        return AGSKIndividual(self.x - individual.x)

    def __mul__(self, num):
        return AGSKIndividual(self.x * num)

    def __lt__(self, individual):
        return list(self.x) < list(individual.x)

def AGSK(dim, MAX_FES, fun):
    G = 0
    NP = 20 * dim
    GEN = 100
    objs = list()

    P = [AGSKIndividual(np.random.rand(dim)) for _ in range(NP)]
    for i in range(NP):
            P[i].objective = evaluate(P[i], fun)
    g_best = P[0]
    # 5
    for G in range(GEN):
        # 6
        dim_junior = int(dim * (1 - G/GEN) ** k_r)
        dim_senior = dim - dim_junior
        #7
        P_junior = JuniorGSK(NP, dim_junior, P)

        #8
        P_senior = SeniorGSK(NP, dim_junior, dim_senior, P_junior)

        #9
        #P_junior.extend(P_senior)
        new_P = P_senior
        for i in range(NP):
            new_P[i].objective = evaluate(new_P[i], fun)

            if new_P[i].objective < P[i].objective:
                P[i] = new_P[i]

        #10
        P = sort_best(P)
        best = P[0]

        if best.objective < g_best.objective:
            g_best = best

        print(g_best)
        objs.append(g_best.objective)

    plt.plot(range(len(objs)), objs)
    plt.show()

def JuniorGSK(NP, dim, P):
    # jest posortowewane best -> wosrt
    new_P = copy.deepcopy(P)
    for i in range(NP):
        x_r = np.random.choice(P)
        x_i = new_P[i]
        if i == 0:
            x_better = P[i+1]
            x_worse = P[i+2]
        elif i == NP - 1:
            x_better = P[i-1]
            x_worse = P[i-2]
        else:
            x_better = P[i-1]
            x_worse = P[i+1]
        x_i_new = x_i
        for j in range(dim):
            if np.random.rand() <= k_r:
                if x_i.objective > x_r.objective:
                    x_i_new.x[j] = x_i.x[j] + ((x_better.x[j] - x_worse.x[j]) + (x_r.x[j] - x_i.x[j])) * k_f
                else:
                    x_i_new.x[j] = x_i.x[j] + ((x_better.x[j] - x_worse.x[j]) + (x_i.x[j] - x_r.x[j])) * k_f
            else:
                x_i_new.x[j] = x_i.x[j]
        new_P[i] = x_i_new
    return new_P

def SeniorGSK(NP, dim_junior, dim_senior, P):
    x_p_best = P[:int(NP * p)]
    x_p_worst = P[int(NP * p):int(NP - (2 * p))]
    x_mid = P[int(NP - 2 * p):]
    new_P = copy.deepcopy(P)
    for i in range(NP):
        x_i = new_P[i]
        x_i_new = x_i
        x_pb = np.random.choice(x_p_best)
        x_pw = np.random.choice(x_p_worst)
        x_m = np.random.choice(x_mid)
        for j in range(dim_junior, dim_senior):
            if np.random.rand() <= k_r:
                if x_i.objective > x_m.objective:
                    x_i_new.x[j] = x_i.x[j] + k_f * ((x_pb.x[j] - x_pw.x[j]) + (x_m.x[j] - x_i.x[j]))
                else:
                    x_i_new.x[j] = x_i.x[j] + k_f * ((x_pb.x[j] - x_pw.x[j]) + (x_i.x[j] - x_m.x[j]))
            else:
                x_i_new.x[j] = x_i.x[j]
        new_P[i] = x_i_new
    return new_P


def evaluate(x, fun):
    return fun(x.x)

def sort_best(P):
    return sorted(P, key=lambda x: x.objective)

# TODO
# Control Adaptive setting
# LPSR