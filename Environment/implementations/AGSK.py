import numpy as np
import copy
import matplotlib.pyplot as plt
from implementations.evolutionary_algorithm import EvolutionaryAlgorithm, Individual

class AGSK(EvolutionaryAlgorithm):
    def __init__(self, k_r=0.5, k_f=0.5, k=0.7, p=0.05, t_MAX=1000):
        super().__init__()
        self.k_r = k_r
        self.k_f = k_f
        self.k = k
        self.p = p
        self.t_MAX = t_MAX

    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.NP = 20 * self.D

    def initialize_population(self):
        self.P = [AGSKIndividual(np.random.rand(self.D)) for _ in range(self.NP)]

    def evaluate_population(self):
        for i in range(self.NP):
            self.evaluate_individual(self.P[i])

    def before_start(self):
        self.P = sorted(self.P, key=lambda x: x.objective)
        self.global_best = self.P[0]

    def prepare_to_generate_population(self):
        self.D_junior = int(self.D * (1 - self.t/self.t_MAX) ** self.k_r)
        self.D_senior = self.D - self.D_junior

        self.P = sorted(self.P, key=lambda x: x.objective)
        self.global_best = self.P[0]

    def mutation(self):
        self.T = list()

        x_p_best = self.P[:int(self.NP * self.p)]
        x_p_worst = self.P[int(self.NP * self.p):int(self.NP - (2 * self.p))]
        x_mid = self.P[int(self.NP - 2 * self.p):]

        for i in range(self.NP):
            # junior phase
            x_r = np.random.choice(self.P)
            x_i = self.P[i]

            if i == 0:
                x_better = self.P[i+1]
                x_worse = self.P[i+2]
            elif i == self.NP - 1:
                x_better = self.P[i-1]
                x_worse = self.P[i-2]
            else:
                x_better = self.P[i-1]
                x_worse = self.P[i+1]
            x_i_new = x_i
            for j in range(self.D_junior):
                if x_i.objective > x_r.objective:
                    x_i_new.x[j] = x_i.x[j] + ((x_better.x[j] - x_worse.x[j]) + (x_r.x[j] - x_i.x[j])) * self.k_f
                else:
                    x_i_new.x[j] = x_i.x[j] + ((x_better.x[j] - x_worse.x[j]) + (x_i.x[j] - x_r.x[j])) * self.k_f

            # senior phase
            x_pb = np.random.choice(x_p_best)
            x_pw = np.random.choice(x_p_worst)
            x_m = np.random.choice(x_mid)
            for j in range(self.D_junior, self.D_senior):
                if x_i.objective > x_m.objective:
                    x_i_new.x[j] = x_i.x[j] + ((x_pb.x[j] - x_pw.x[j]) + (x_m.x[j] - x_i.x[j])) * self.k_f
                else:
                    x_i_new.x[j] = x_i.x[j] + ((x_pb.x[j] - x_pw.x[j]) + (x_i.x[j] - x_m.x[j])) * self.k_f

            self.T.append(x_i_new)

    def crossover(self):
        self.O = list()
        for i in range(self.NP):
            x = self.P[i]
            u = self.T[i]
            for j in range(self.D):
                if np.random.rand() < self.k_r:
                    x.x[j] = u.x[j]
            self.O.append(x)

        self.new_P = self.O

    def operation_after_generate(self):
        for i in range(self.NP):
            if self.new_P[i].objective < self.P[i].objective:
                self.P[i] = self.new_P[i]

        self.get_pbest()
        self.FES += self.NP

        if self.FES >= self.MAX_FES:
            self.stop = True

        self.FESs.append(self.FES)
        self.bests_values.append(self.global_best.objective)

    def get_pbest(self):
        best = sorted(self.P, key=lambda x: x.objective)
        ind = int(np.round(self.p * np.size(self.P)))
        ind = max(ind, 1)
        self.pbest = best[:ind]
        self.global_best = self.pbest[0]
        

class AGSKIndividual(Individual):
    def __init__(self, x):
        super().__init__(x)

# TODO
# Control Adaptive setting
# LPSR