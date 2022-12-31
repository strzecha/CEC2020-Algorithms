from implementations.j2020 import j2020Algorithm
from implementations.IMODE_2 import IMODE
from test_functions.test_functions import (fun1D, fun2D, fun3D, 
                                            hypersphere2D, easom2D, 
                                            schwefel2D, himmelblau2D)
from utils.visualization import draw_graph_evaluations



alg_1 = j2020Algorithm(50, 10)
alg_2 = IMODE()

print("Fun1D:")
(best_sol_1, objective_val_1) = alg_1.optimize(fun1D, 1, 1000, 100, -100)
(best_sol_2, objective_val_2) = alg_2.optimize(fun1D, 1, 1000, 100, -100)
print(best_sol_1, objective_val_1)
print(best_sol_2, objective_val_2)

draw_graph_evaluations([(alg_1, "j2020"), (alg_2, "IMODE"), (alg_2, "IMODE")])
"""
print(20 * "-")
print("Fun2D:")
(best_sol, objective_val) = alg.optimize(fun2D, 2, 2000, 100, -100)
print(best_sol, objective_val)

draw_graph_evaluations([(alg, "j2020")])

print(20 * "-")
print("Fun3D:")
(best_sol, objective_val) = alg.optimize(fun3D, 3, 5000, 100, -100)
print(best_sol, objective_val)

draw_graph_evaluations([(alg, "j2020")])

print(20 * "-")
print("Hypersphere 2D:")
(best_sol, objective_val) = alg.optimize(hypersphere2D, 2, 5000, 100, -100)
print(best_sol, objective_val)

draw_graph_evaluations([(alg, "j2020")])

print(20 * "-")
print("Easom 2D:")
(best_sol, objective_val) = alg.optimize(easom2D, 2, 5000, 100, -100)
print(best_sol, objective_val)

draw_graph_evaluations([(alg, "j2020")])

print(20 * "-")
print("Schwefel 2D:")
(best_sol, objective_val) = alg.optimize(schwefel2D, 2, 5000, 100, -100)
print(best_sol, objective_val)

draw_graph_evaluations([(alg, "j2020")])

print(20 * "-")
print("Himmelblau 2D:")
(best_sol, objective_val) = alg.optimize(himmelblau2D, 2, 5000, 100, -100)
print(best_sol, objective_val)

draw_graph_evaluations([(alg, "j2020")])

"""
