import numpy as np
from copy import deepcopy

from implementations.basic_algorithms.DE import DE
from implementations.basic_algorithms.BCHM import wrapping


INITIAL_EQUALITY_PERCENT = 0.25
INITIAL_TOLERANCE = 1e-16
INITIAL_F = 0.5
INITIAL_CR = 0.9
INITIAL_F_LOWER_BIG = 0.01
INITIAL_F_LOWER_SMALL = 0.17
INITIAL_F_UPPER = 1.1
INITIAL_CR_LOWER_BIG = 0.0
INITIAL_CR_LOWER_SMALL = 0.0
INITIAL_CR_UPPER_BIG = 1.0
INITIAL_CR_UPPER_SMALL = 0.7
INITIAL_CR_ADAPTATION_PROBABILITY = 0.1
INITIAL_F_ADAPTATION_PROBABILITY = 0.1


class j2020(DE):
    def __init__(self):
        super().__init__()
        self.name = "j2020"

    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)

        self.sNP = max(self.D, 4)
        self.bNP = self.sNP * 7 
        self.NP = self.bNP + self.sNP

        if self.gn + self.hn > 0:
            self.svc_default = np.inf
        else:
            self.svc_default = 0 

        # parameters to reinitialize populations
        self.my_eqs = INITIAL_EQUALITY_PERCENT
        self.eps = INITIAL_TOLERANCE
        self.age = 0

        # parameters to adapt F and CR
        self.F_init = INITIAL_F
        self.CR_init = INITIAL_CR
        self.Flb = INITIAL_F_LOWER_BIG
        self.Fls = INITIAL_F_LOWER_SMALL        
        self.Fu = INITIAL_F_UPPER 
        self.CRlb = INITIAL_CR_LOWER_BIG
        self.CRub = INITIAL_CR_UPPER_BIG
        self.CRus = INITIAL_CR_UPPER_SMALL 
        self.CRls = INITIAL_CR_LOWER_SMALL
        self.tao1 = INITIAL_F_ADAPTATION_PROBABILITY
        self.tao2 = INITIAL_CR_ADAPTATION_PROBABILITY 

    def initialize_population(self):
        self.P_b = np.array([self.individual_generic(self.MIN + np.random.rand(self.D) * (self.MAX - self.MIN),
                                            F=self.F_init, CR=self.CR_init) for i in range(self.bNP)])
        self.P_s = np.array([self.individual_generic(self.MIN + np.random.rand(self.D) * (self.MAX - self.MIN),
                                            F=self.F_init, CR=self.CR_init) for i in range(self.sNP)])    

    def evaluate_initial_population(self):
        for i in range(self.bNP):
            self._evaluate_individual(self.P_b[i])
            self.P_b[i].svc = self._get_svc(self.P_b[i].g, self.P_b[i].h)
            if i == 0 or self._is_better(self.P_b[i], self.global_best):
                self.global_best_index = i
                self.global_best = deepcopy(self.P_b[i])
            self.FES += 1

        for i in range(self.sNP):
            self._evaluate_individual(self.P_s[i])
            self.P_s[i].svc = self._get_svc(self.P_s[i].g, self.P_s[i].h)
            if self._is_better(self.P_s[i], self.global_best):
                self.global_best_index = i + self.bNP
                self.global_best = deepcopy(self.P_s[i])
            self.FES += 1

    def before_start(self):
        super().before_start()
        self.global_best_index = 0
        self.global_best = self.individual_generic(objective=np.inf, svc=self.svc_default)
        self.big_population = True
        self.current_individual_index = 0
        self.m = 0

    def prepare_to_generate(self):
        if self.big_population:
            if self.current_individual_index == self.bNP:
                self.big_population = False
                self.current_individual_index = 0
        else:
            if self.current_individual_index == self.sNP:
                self.current_individual_index = 0
                self.m += 1
                if self.m == 7:
                    self.big_population = True
                    self.m = 0

        if self.current_individual_index == 0 and self.big_population:
            self._reinitialize_big_population()
            self._reinitialize_small_population()

        if self.current_individual_index == 0 and not self.big_population and self.global_best_index < self.bNP:
            self.P_s[0].x, self.P_s[0].objective = deepcopy(self.P_b[self.global_best_index].x), self.P_b[self.global_best_index].objective
            self.global_best_index = self.bNP
    
    def mutation(self):
        r1, r2, r3 = self._get_random_indexes()
        F, CR = self._generate_parameters()
        self.O.F = F
        self.O.CR = CR

        self.T = self.individual_generic(np.zeros(self.D))
        if r1 < self.bNP:
            xr1 = self.P_b[r1]
        else:
            xr1 = self.P_s[r1 % self.bNP]

        if r2 < self.bNP:
            xr2 = self.P_b[r2]
        else:
            xr2 = self.P_s[r2 % self.bNP]

        if r3 < self.bNP:
            xr3 = self.P_b[r3]
        else:
            xr3 = self.P_s[r3 % self.bNP]

        self.T = xr1 + self.O.F * (xr2 - xr3)

    def repair_boundary_constraints(self):
        self.T = wrapping(self.T, self.D, self.MIN, self.MAX)

    def crossover(self):
        jrand = np.random.randint(0, self.D)
        for j in range(self.D):
            if np.random.rand() < self.O.CR or j == jrand:
                self.O.x[j] = self.T.x[j]

    def evaluate_new_population(self):
        self._evaluate_individual(self.O)
        self.O.svc = self._get_svc(self.O.g, self.O.h)
        self.FES += 1

    def crowding(self, U):
        min_dist = np.linalg.norm(self.P_b[0].x - U.x)
        index = 0
        for i in range(1, self.bNP):
            dist = np.linalg.norm(self.P_b[i].x - U.x)
            if dist < min_dist:
                min_dist = dist
                index = i
        return index
    
    def selection(self):
        index = self.current_individual_index           
            
        if self.big_population: 
            self.age += 1
            index = self.crowding(self.O)
            if self._is_better(self.O, self.P_b[index]):
                self.P_b[index] = deepcopy(self.O)
                if self._is_better(self.O, self.global_best):
                    self.age = 0
                    self.global_best_index = index
                    self.global_best = deepcopy(self.P_b[self.global_best_index])
        else:
            if self._is_better(self.O, self.P_s[index]):
                self.P_s[index] = deepcopy(self.O)
                if self._is_better(self.O, self.global_best):
                    self.global_best_index = index + self.bNP
                    self.global_best = deepcopy(self.P_s[self.global_best_index % self.bNP])
    
    def after_generate(self):
        super().after_generate()
        self.current_individual_index += 1

    # helpful methods
    def _is_better(self, x1, x2):
        return ((x1.svc == 0) & (x2.svc == 0) & (x1.objective <= x2.objective)) | (x1.svc < x2.svc)
    
    def _is_similar_small_population(self, best_objective):
        eqs = 0
        for i in range(self.sNP):
            if abs(self.P_s[i].objective - best_objective) < self.eps:
                eqs += 1
        return eqs > np.ceil(self.my_eqs * self.sNP)
    
    def _is_similar_big_population(self):
        P = sorted(self.P_b, key=lambda x: x.objective)
        first = 0
        last = np.ceil(self.my_eqs * self.bNP).astype(int) - 1
        
        if np.abs(P[first].objective - P[last].objective) < self.eps:
            return True
            
        return False
    
    def _reinitialize_big_population(self):
        if self._is_similar_big_population() or self.age > self.FES_MAX / 10:
            self.P_b = np.array([self.individual_generic(self.MIN + np.random.rand(self.D) * (self.MAX - self.MIN),
                                        float('inf'), F=self.F_init, CR=self.CR_init, svc=self.svc_default) for i in range(self.bNP)])
            self.age = 0
            self.global_best_index = self.bNP
            self.global_best = deepcopy(self.P_s[0])
            for w in range(self.sNP):
                if self._is_better(self.P_s[w], self.global_best):
                    self.global_best_index = w + self.bNP
                    self.global_best = deepcopy(self.P_s[w])

    def _reinitialize_small_population(self):
        best_objective = np.min(np.array([p.objective for p in self.P_s]))
        if self._is_similar_small_population(best_objective):
            for w in range(self.sNP):
                if self.P_s[w].objective == best_objective:
                    continue
                self.P_s[w] = self.individual_generic(self.MIN + np.random.rand(self.D) * (self.MAX - self.MIN),
                                            float('inf'), F=self.F_init, CR=self.CR_init, svc=self.svc_default)

    def _generate_parameters(self):
        if self.big_population:
            Fl = self.Flb
            CRl = self.CRlb
            CRu = self.CRub
        else:
            Fl = self.Fls
            CRl = self.CRls
            CRu = self.CRus
        
        if not self.big_population:
            self.O = deepcopy(self.P_s[self.current_individual_index])
        else:
            self.O = deepcopy(self.P_b[self.current_individual_index])

        if np.random.rand() < self.tao1:
            F = Fl + np.random.rand() * self.Fu
        else:
            F = self.O.F
        if np.random.rand() < self.tao2:
            CR = CRl + np.random.rand() * CRu
        else:
            CR = self.O.CR

        return F, CR

    def _get_random_indexes(self):
        Ms = 0
        if self.big_population:
            if self.FES < self.FES_MAX/3:
                Ms = 1
            elif self.FES < 2*self.FES_MAX/3:
                Ms = 2
            else:
                Ms = 3

            NP = self.bNP
        else:
            NP = self.sNP
            
        new_arr = np.delete(np.arange(NP), self.current_individual_index)
        r1 = np.random.choice(new_arr)

        new_arr = np.delete(np.arange(NP + Ms), (self.current_individual_index, r1))
        r2 = np.random.choice(new_arr)

        new_arr = np.delete(np.arange(NP + Ms), (self.current_individual_index, r1, r2))
        r3 = np.random.choice(new_arr)
        
        if not self.big_population:
            r1 += self.bNP
            r2 += self.bNP
            r3 += self.bNP

        return r1, r2, r3

    
