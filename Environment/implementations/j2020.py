import numpy as np

from implementations.evolutionary_algorithm import EvolutionaryAlgorithm, Individual

class j2020Individual(Individual):
    def __init__(self, x, F, CR):
        super().__init__(x)
        self.F = F
        self.CR = CR

    def __add__(self, individual):
        return j2020Individual(self.x + individual.x, self.F, self.CR)

    def __sub__(self, individual):
        return j2020Individual(self.x - individual.x, self.F, self.CR)

    def __mul__(self, num):
        return j2020Individual(self.x * num, self.F, self.CR)

class j2020(EvolutionaryAlgorithm):
    def __init__(self, bNP=None, sNP=None, F=0.5, Flb=0.01, Fls=0.17, Fu=1.1, tau1=0.1, CR=0.9, tau2=0.1, eps=10**(-16), myEqs=25):
        self.bNP = bNP
        self.sNP = sNP
        self.m = 0
        self.noImprove = 0
        self.Flb = Flb
        self.Fls = Fls
        self.Fu = Fu
        self.eps = eps
        self.myEqs = myEqs
        self.ageLmt = 0
        self.tau1 = tau1
        self.tau2 = tau2
        self.F = F
        self.CR = CR

    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.bNP = self.bNP if self.bNP else 7 * self.D
        self.sNP = self.sNP if self.sNP else self.D

        self.MIN_SNP = 3 # various individuals in mutation

        self.sNP = max(self.MIN_SNP, self.sNP)

        self.m = self.bNP // self.sNP

        self.ageLmt = self.MAX_FES // 10
        self.noImprove = 0


    def initialize_population(self):
        self.Pb = np.array([j2020Individual(np.random.uniform(self.MIN, self.MAX, self.D), self.F, self.CR) for _ in range(self.bNP)])
        self.Ps = np.array([j2020Individual(np.random.uniform(self.MIN, self.MAX, self.D), self.F, self.CR) for _ in range(self.sNP)])

    def evaluate_initial_population(self):
        for i in range(self.bNP):
            self.evaluate_individual(self.Pb[i])
        for i in range(self.sNP):
            self.evaluate_individual(self.Ps[i])

        self.FES += self.bNP + self.sNP
        self.global_best = self.get_best(self.Ps)

        self.FESs.append(self.FES)
        self.bests_values.append(self.global_best.objective)

    def evaluate_new_population(self):
        for i in range(self.bNP):
            self.evaluate_individual(self.O[i])

        self.FES += self.bNP

    def get_best(self, P):
        new_P = sorted(P, key=lambda x: x.objective)
        return new_P[0]

    def before_start(self):
        return super().before_start()

    def prepare_to_generate_population(self):
        self.check_reinitialize_big()
        self.check_reinitialize_small()

        self.Ms_size = 1
        if self.FES > self.MAX_FES / 3:
            self.Ms_size = 2
        elif self.FES > self.MAX_FES * 2 / 3:
            self.Ms_size = 3

    def mutation(self):
        self.T = np.array([])
        for x in self.Pb:
            Ms = np.array([self.Ps[i] for i in np.random.randint(0, self.sNP, self.Ms_size)])
            v = self.mutation_big(x, Ms)

            self.T = np.append(self.T, v)

    def crossover(self):
        self.O = np.array([])
        for i in range(self.bNP):
            x = self.Pb[i]
            v = self.T[i]
            u = self.do_crossover(x, v)
            self.O = np.append(self.O, u)

    def selection(self):
        self.new_Pb = np.array([])

        for i in range(self.bNP):
            
            u = self.O[i]
            closest_individual_index = self.crowding(u)
            x = self.Pb[closest_individual_index]

            if u.objective <= x.objective:
                xi = u # x, F, CR
            else:
                xi = x # x, F, CR

            self.new_Pb = np.append(self.Pb, xi)

        self.Pb = self.new_Pb

    def after_generate(self):
        best_Pb = self.get_best(self.Pb)
        self.check_if_improve(best_Pb)

        for k in range(self.m):
            new_Ps = np.array([])

            for x in self.Ps:
                v = self.mutation_small(x)
                u = self.do_crossover(x, v)
                self.evaluate_individual(u)
                if u.objective <= x.objective:
                    xi = u # x, F, CR
                else:
                    xi = x # x, F, CR
                new_Ps = np.append(new_Ps, xi)

            self.FES += self.sNP

            self.Ps = new_Ps

        self.global_best = self.get_best(self.Ps)

        self.bests_values.append(self.global_best.objective)
        self.FESs.append(self.FES)
        if self.FES >= self.MAX_FES:
            self.stop = True

    def mutation_big(self, x, Ms):
        return self.do_mutation(x, self.Pb, self.Flb, Ms)

    def mutation_small(self, x):
        return self.do_mutation(x, self.Ps, self.Fls)

    def do_mutation(self, x, P, Fl, Ms=None):
        # jDE mutation
        F = Fl + np.random.random() * self.Fu if np.random.random() < self.tau1 else x.F
        r1 = r2 = r3 = 0
        
        new_P = np.concatenate((P, Ms)) if Ms is not None else P

        while r1 == r2 or r2 == r3 or r1 == r3:
            r1 = np.random.randint(0, P.shape[0])
            r2 = np.random.randint(0, new_P.shape[0])
            r3 = np.random.randint(0, new_P.shape[0])

        v = new_P[r1] + (new_P[r2] - new_P[r3]) * F
        v.F = F

        return v

    def crowding(self, u):
        dist = np.linalg.norm(self.Pb[0].x - u.x)
        min_dist = dist
        min_index = 0

        for i in range(1, self.bNP):
            dist = np.linalg.norm(self.Pb[i].x - u.x)
            if dist < min_dist:
                min_dist = dist
                min_index = i

        return min_index

    def do_crossover(self, x, v):
        # jDE crossover
        CR = np.random.random() if np.random.random() < self.tau2 else x.CR
        
        u = j2020Individual(v.x, v.F, CR)
        jrand = np.random.randint(0, self.D - 1) if self.D > 1 else 0

        for j in range(self.D):
            if np.random.random() <= CR or j == jrand:
                u.x[j] = v.x[j]

                while u.x[j] < self.MIN[j]:
                    u.x[j] += (self.MAX[j] - self.MIN[j])
                while u.x[j] > self.MAX[j]:
                    u.x[j] -= (self.MAX[j] - self.MIN[j])
            else:
                u.x[j] = x.x[j]
        return u

    def check_if_improve(self, current_best):
        if current_best.objective < self.global_best.objective: 
            self.Ps[self.FES % self.Ps.shape[0]] = current_best
            self.noImprove = 0
        else:
            self.noImprove += 1

    def check_reinitialize_big(self):
        self.Pb = np.array(sorted(self.Pb, key=lambda ind: ind.objective))

        self.Pb = self.check_reinitialize(self.Pb, self.noImprove >= self.ageLmt)

    def check_reinitialize_small(self):
        self.Ps = np.array(sorted(self.Ps, key=lambda ind: ind.objective))
        best = self.Ps[0]

        self.Ps = self.check_reinitialize(self.Ps)
        self.Ps[-1] = best

    def check_reinitialize(self, P, statement=True):
        first = 0
        last = int(P.shape[0] * self.myEqs / 100)

        dif = P[first].objective - P[last].objective

        if abs(dif) < self.eps and statement:
            new_P = np.array([j2020Individual(np.random.uniform(self.MIN, self.MAX, self.D), P[i].F, P[i].CR) for i in range(P.shape[0])])
            for i in range(P.shape[0]):
                self.evaluate_individual(new_P[i])
            P = new_P

        return P
