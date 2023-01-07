from implementations.j2020 import j2020
from implementations.IMODE import IMODE
from implementations.EnMODE import EnMODE
from implementations.COLSHADE import COLSHADE
from implementations.AGSK import AGSK
from implementations.esCMAgES import esCMAgES
from test_functions.test_functions import (fun1D, fun2D, fun3D, 
                                            hypersphere2D, easom2D, 
                                            schwefel2D, himmelblau2D, hypersphere5D, fun2Dconst, fun2Dblank)
from utils.visualization import draw_graph_evaluations

alg_1 = esCMAgES()
NAME = "EnMODE"
total = 0

print("fun1D")
dim = 1
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(fun1D, 1, 1000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

print("fun2D")
dim = 2
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(fun2D, 2, 2000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

print("hypersphere5D")
dim = 5
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(hypersphere5D, 5, 10000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

"""

for i in range(25):
    print("e2D:")
    dim = 2
    MAX = [100] * dim
    MIN = [-100] * dim
    (best_sol_1, objective_val_1) = alg_1.optimize(fun2Dconst, 2, 10000, MAX, MIN)
    print(best_sol_1, objective_val_1)
    total += objective_val_1
    #draw_graph_evaluations([ (alg_1, NAME)])

    print("e2D:")
    dim = 2
    MAX = [100] * dim
    MIN = [-100] * dim
    (best_sol_1, objective_val_1) = alg_1.optimize(fun2Dblank, 2, 10000, MAX, MIN)
    print(best_sol_1, objective_val_1)
    #draw_graph_evaluations([ (alg_1, NAME)])


print(total / 25)
"""
"""
print("Fun1D:")
dim = 1
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(fun1D, 1, 1000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

print("Fun2D:")
dim = 2
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(fun2D, 2, 5000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

print("Himmelblau2D:")
dim = 2
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(himmelblau2D, 2, 5000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

print("Schweffel2D:")
dim = 2
MAX = [500] * dim
MIN = [-500] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(schwefel2D, 2, 5000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

print("Hypersphere5D:")
dim = 5
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(hypersphere5D, 5, 5000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])
"""