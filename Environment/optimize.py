import numpy as np
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.data_reader import get_properties

from implementations.AGSK import AGSK
from implementations.IMODE import IMODE
from implementations.EnMODE import EnMODE
from implementations.COLSHADE import COLSHADE
from implementations.j2020 import j2020
from implementations.esCMAgES import esCMAgES

from test_functions.cec2020_constrained import CEC2020_CONSTRAINED
from test_functions.cec2020_nonconstrained import (CEC2020_NONCONSTRAINED_5D, CEC2020_NONCONSTRAINED_10D, 
                                                   CEC2020_NONCONSTRAINED_15D, CEC2020_NONCONSTRAINED_20D)

from utils.optimization import perform_optimization
from utils.cec20_parameters import initialize_cec2020_constrained_parameters, initialize_cec2020_nonconstrained_parameters

def main():
    properties = get_properties()

    BENCHMARKS_CONSTRAINED = properties.get("BENCHMARKS_CONSTRAINED").data.split(",")
    BENCHMARKS_NONCONSTRAINED = properties.get("BENCHMARKS_NONCONSTRAINED").data.split(",")
    BENCHMARKS_NAMES = BENCHMARKS_CONSTRAINED + BENCHMARKS_NONCONSTRAINED
    ALGORITHMS_NAMES = properties.get("ALGORITHMS").data.split(",")
    RESULTS_DIR = properties.get("RESULTS_DIR").data

    parser = argparse.ArgumentParser()

    parser.add_argument('benchmark', choices=BENCHMARKS_NAMES)
    parser.add_argument('algorithm', choices=ALGORITHMS_NAMES)
    parser.add_argument('function', type=int)
    parser.add_argument('dimensionality', type=int)
    parser.add_argument('runs', type=int)
    parser.add_argument('-f', '--fes', type=int, required=False)
    parser.add_argument('-p', '--path', type=str)

    args = parser.parse_args()

    if args.path:
        root = args.path
    else:
        root = RESULTS_DIR

    if args.algorithm == 'j2020':
        alg = j2020()
    elif args.algorithm == 'AGSK':
        alg = AGSK()
    elif args.algorithm == 'IMODE':
        alg = IMODE()
    elif args.algorithm == 'EnMODE':
        alg = EnMODE()
    elif args.algorithm == 'COLSHADE':
        alg = COLSHADE()
    elif args.algorithm == 'esCMAgES':
        alg = esCMAgES()


    if args.benchmark == 'CEC20nonconstr':
        if args.dimensionality == 5:
            fun = CEC2020_NONCONSTRAINED_5D[args.function - 1]
        elif args.dimensionality == 10:
            fun = CEC2020_NONCONSTRAINED_10D[args.function - 1]    
        elif args.dimensionality == 15:
            fun = CEC2020_NONCONSTRAINED_15D[args.function - 1]  
        elif args.dimensionality == 20:
            fun = CEC2020_NONCONSTRAINED_20D[args.function - 1] 

        FES_MAX, MIN, MAX = initialize_cec2020_nonconstrained_parameters(args.dimensionality)
        D = args.dimensionality

    elif args.benchmark == 'CEC20constr':
        fun = CEC2020_CONSTRAINED[args.function - 1]

        FES_MAX, MIN, MAX, D = initialize_cec2020_constrained_parameters(args.function)

    if args.fes:
        FES_MAX = args.fes

    path  =f"{root}/{args.algorithm}/{args.benchmark}"
    file_name = f"F{args.function}-{D}D"
    start = time.time()
    perform_optimization(alg, args.benchmark, fun, path, file_name,
                        FES_MAX, MIN, MAX, args.runs)
    stop = start - time.time()
    print(stop, "s")

if __name__ == "__main__":
    main()