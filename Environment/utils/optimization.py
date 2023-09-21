import csv
import numpy as np
import os

from random_generator import random_generator
from utils.data_reader import get_properties

properties = get_properties()

RESULT_LABEL = properties.get("RESULT_LABEL").data
ERROR_LABEL = properties.get("ERROR_LABEL").data
FES_CONSTRAINT_LABEL = properties.get("FES_CONSTRAINT_LABEL").data
FES_SUCCESS_LABEL = properties.get("FES_SUCCESS_LABEL").data
CONSTRAINT_VIOLATION_LABEL = properties.get("CONSTRAINT_VIOLATION_LABEL").data
MEAN_CONSTRAINT_VIOLATION_LABEL = properties.get("MEAN_CONSTRAINT_VIOLATION_LABEL").data

def get_mean_violation_constraints(fun, x, equality_tolerance):
    _, g, h = fun(x.reshape(1, np.size(x)))
    svc = np.sum(np.maximum(g, 0)) + np.sum(np.maximum(np.abs(h) - equality_tolerance, 0))

    constraints_num = max(fun.inequality_constraints_num + fun.equality_constraints_num, 1)

    return svc / constraints_num


def perform_optimization(alg, benchmark, fun, path, filename, FES, MIN, MAX, runs):
    os.makedirs(path, exist_ok=True)

    dimensionality = fun.dimensionality
    
    csvfile = open(f"{path}/{filename}.csv", 'w', newline='')
    fieldnames = ["No.", RESULT_LABEL, ERROR_LABEL, CONSTRAINT_VIOLATION_LABEL, 
                  MEAN_CONSTRAINT_VIOLATION_LABEL, FES_CONSTRAINT_LABEL, FES_SUCCESS_LABEL]
    spamwriter = csv.DictWriter(csvfile, delimiter=';', quotechar="|", quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
    spamwriter.writeheader()

    print(f"Benchmark: {benchmark}")
    print(f"Problem: {fun.ID}")
    print(f"Algorithm: {alg.name}")
    for run in range(runs):
        #random_generator.restart_generator()
        #np.random.rand((run + 1) * 99)
        best_ind, FES_feasible, FES_reached, bests = alg.optimize(fun, dimensionality, FES, MAX, MIN)
        mean_vio = get_mean_violation_constraints(fun, best_ind.x, 1e-4)
        print(f"Run {run+1}. best = {best_ind.objective} mean constraints violation = {mean_vio} FES: {FES_reached}, FES feasible: {FES_feasible}")

        if fun.global_minimum is not None:
            error = np.abs(best_ind.objective - fun.global_minimum)
        else:
            error = "undefined"
        
        spamwriter.writerow({"No." : run+1, RESULT_LABEL : best_ind.objective, ERROR_LABEL : error,
                                CONSTRAINT_VIOLATION_LABEL : best_ind.svc, MEAN_CONSTRAINT_VIOLATION_LABEL : mean_vio, 
                                FES_CONSTRAINT_LABEL : FES_feasible, FES_SUCCESS_LABEL : FES_reached})
        

        dir = f"{path}/{filename}-runs"
        os.makedirs(dir, exist_ok=True)
        name = f"{dir}/{run+1}"
        np.save(name, bests[1:])
        
