import numpy as np
import copy
from implementations.evolutionary_algorithm import EvolutionaryAlgorithm, Individual

class AGSK(EvolutionaryAlgorithm):
    def __init__(self, k=0.7, p=0.05, t_MAX=1000, c=0.05, NP_min=12):
        super().__init__()
        self.k = k
        self.p = p
        self.t_MAX = t_MAX
        self.c = c
        self.Kw_P = np.array([0.85, 0.05, 0.05, 0.05])
        self.pairs = [(0.1, 0.2), (1.0, 0.1), (0.5, 0.9), (1.0,  0.9)]
        self.omegas = np.array([])
        self.NP_min = NP_min

    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.NP = 20 * self.D
        self.NP_init = self.NP

    def initialize_population(self):
        self.P = [AGSKIndividual(np.random.rand(self.D)) for _ in range(self.NP)]

    def evaluate_population(self):
        for i in range(self.NP):
            self.evaluate_individual(self.P[i])

        self.FES += self.NP

    def before_start(self):
        self.get_best()

    def update_Kw_P(self):
        omega_ps = np.sum([self.P[i].objective - self.old_P[i].objective for i in range(self.NP)])
        self.omegas = np.append(self.omegas, omega_ps)

        delta_ps = np.max([0.05, omega_ps / np.sum(self.omegas)])

        self.Kw_P = (1 - self.c) * self.Kw_P + self.c * delta_ps

    def adapt_parameters(self):
        for i in range(self.NP):
            r = np.random.rand()

            total_prob = 0
            for prob, pair in zip(self.Kw_P, self.pairs):
                if r < total_prob + prob:
                    self.P[i].F, self.P[i].CR = pair
                    break
                total_prob += prob


    def prepare_to_generate_population(self):
        self.D_junior = int(self.D * (1 - self.t/self.t_MAX) ** self.k)
        self.D_senior = self.D

        self.get_best()

        if self.FES > 0.1 * self.MAX_FES:
            self.update_Kw_P()

        self.adapt_parameters()

        self.old_P = copy.deepcopy(self.P)

    def mutation(self):
        self.T = list()
        x_p_index = max(int(self.NP * self.p), 1)
        x_p_best = self.P[:x_p_index]
        x_mid = self.P[x_p_index:-x_p_index]
        x_p_worst = self.P[-x_p_index:]

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
            x_i_new = copy.deepcopy(x_i)
            for j in range(self.D_junior):
                if x_i.objective > x_r.objective:
                    x_i_new.x[j] = x_i.x[j] + ((x_better.x[j] - x_worse.x[j]) + (x_r.x[j] - x_i.x[j])) * x_i.F
                else:
                    x_i_new.x[j] = x_i.x[j] + ((x_better.x[j] - x_worse.x[j]) + (x_i.x[j] - x_r.x[j])) * x_i.F

            # senior phase
            x_pb = np.random.choice(x_p_best)
            x_pw = np.random.choice(x_p_worst)
            x_m = np.random.choice(x_mid)
            for j in range(self.D_junior, self.D_senior):
                if x_i.objective > x_m.objective:
                    x_i_new.x[j] = x_i.x[j] + ((x_pb.x[j] - x_pw.x[j]) + (x_m.x[j] - x_i.x[j])) * x_i.F
                else:
                    x_i_new.x[j] = x_i.x[j] + ((x_pb.x[j] - x_pw.x[j]) + (x_i.x[j] - x_m.x[j])) * x_i.F

            self.T.append(x_i_new)

    def crossover(self):
        self.O = list()
        for i in range(self.NP):
            x = copy.deepcopy(self.P[i])
            u = self.T[i]
            for j in range(self.D):
                if np.random.rand() < x.CR:
                    x.x[j] = u.x[j]
            self.O.append(x)

        self.P = self.O

    def selection(self):
        for i in range(self.NP):
            if self.P[i].objective > self.old_P[i].objective:
                self.P[i] = self.old_P[i]

    def after_generate(self):
        self.get_best()
        self.FESs.append(self.FES)
        self.bests_values.append(self.global_best.objective)

        if self.FES >= self.MAX_FES:
            self.stop = True
        
        self.NP = self.LPSR(self.NP_min, self.NP_init, self.MAX_FES, self.FES)
        self.P = self.P[:self.NP]

    def get_best(self):
        self.P = sorted(self.P, key=lambda x: x.objective)
        self.global_best = self.P[0]

class AGSKIndividual(Individual):
    def __init__(self, x):
        super().__init__(x)
        self.CR = 0 # k_r
        self.F = 0 # k_f


