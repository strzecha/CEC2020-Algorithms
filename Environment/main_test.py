from implementations.j2020 import j2020
from implementations.IMODE import IMODE
from implementations.EnMODE import EnMODE
from implementations.COLSHADE import COLSHADE
from implementations.AGSK import AGSK
from test_functions.test_functions import (fun1D, fun2D, fun3D, 
                                            hypersphere2D, easom2D, 
                                            schwefel2D, himmelblau2D, hypersphere5D)
from utils.visualization import draw_graph_evaluations

alg_1 = j2020(50, 10)
NAME = "j2020"


print("Fun1D:")
dim = 1
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(fun1D, 1, 1000, [100], [-100])
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

print("Fun2D:")
dim = 2
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(fun2D, 2, 5000, [100, 100], [-100, -100])
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])

print("Hypersphere5D:")
dim = 5
MAX = [100] * dim
MIN = [-100] * dim
(best_sol_1, objective_val_1) = alg_1.optimize(hypersphere5D, 5, 10000, MAX, MIN)
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, NAME)])