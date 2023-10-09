"""Script to draw ECDF curves
"""
import argparse
import numpy as np

from utils.cec20_parameters import initialize_cec2020_constrained_parameters, initialize_cec2020_nonconstrained_parameters, get_dim_and_constraints_cec20_constrained
from utils.COCO_ECDF import ECDF_per_function, ECDF_group
from utils.data_reader import get_properties
from utils.stats import calc_ECDF_ranking

def main():
    properties = get_properties()

    BENCHMARKS_CONSTRAINED = properties.get("BENCHMARKS_CONSTRAINED").data.split(",")
    BENCHMARKS_NONCONSTRAINED = properties.get("BENCHMARKS_NONCONSTRAINED").data.split(",")
    BENCHMARKS_NAMES = BENCHMARKS_CONSTRAINED + BENCHMARKS_NONCONSTRAINED
    ALGORITHMS_NAMES = properties.get("ALGORITHMS").data.split(",")
    RESULTS_DIR = properties.get("RESULTS_DIR").data
    ECDF_RANKING_DIR = properties.get("ECDF_RANKING_DIR").data

    parser = argparse.ArgumentParser()

    parser.add_argument('benchmark', choices=BENCHMARKS_NAMES)
    parser.add_argument('-a', '--algorithms', choices=ALGORITHMS_NAMES, type=str, nargs='+', required=True)
    parser.add_argument('-f', '--functions', type=int, nargs='*')
    parser.add_argument('-l', '--all', action='store_true')
    parser.add_argument('-d', '--dim', type=int)
    parser.add_argument('-c', "--with_constraints", action='store_true')
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-b', '--budget', type=int)
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('-v', '--save', action='store_true')

    args = parser.parse_args()

    if args.show or args.save:
        if args.path:
            root = args.path
        else:
            root = RESULTS_DIR

        algs = []

        for alg_name in args.algorithms:
            path = f"{root}/{alg_name}/{args.benchmark}"
            alg = [alg_name, path]
            algs.append(alg)

        if args.benchmark == BENCHMARKS_NONCONSTRAINED[0]:
            if not args.dim:
                raise Exception("You must specify dimensionality for CEC20nonconstr benchmark")

            if not args.dim:
                raise Exception("You must specify dimensionality for CEC20nonconstr benchmark")
            if args.all:
                if args.dim == 5:
                    funs = [1, 2, 3, 4, 5, 8, 9, 10]
                else:
                    funs = list(range(1, 11))
            else:
                funs = args.functions

            dims = [args.dim] * len(funs)
            runs = 30
            
            if args.budget:
                budget = args.budget
            else:
                budget, _, _ = initialize_cec2020_nonconstrained_parameters(args.dim)
                budgets = budget * np.ones(len(funs))

        if args.benchmark == BENCHMARKS_CONSTRAINED[0]:
            dims = []
            budgets = []

            if args.all:
                funs = list(range(1, 21))
            else:
                funs = args.functions

            for fun in funs:
                budget, _, _, D = initialize_cec2020_constrained_parameters(fun)
                dims.append(D)
                budgets.append(budget)

            if args.budget:
                budget = args.budget
            runs = 25

        div = 2
        if len(funs) > div:
            rows = len(funs) // div + int(len(funs) % div != 0)
            cols = div
        else:
            rows = 1
            cols = len(funs) 


        dest_path = ECDF_RANKING_DIR
        areas, SR = ECDF_per_function(algs, funs, dims, budgets, runs, rows, cols, save=args.save, show=args.show)
        calc_ECDF_ranking(algs, areas, SR, dest_path, "optimality", args.show, args.save)
        ECDF_group(algs, funs, dims, budgets, runs, save=args.save, show=args.show)

        if args.with_constraints:
            areas, SR = ECDF_per_function(algs, funs, dims, budgets, runs, rows, cols, 
                            optimality=False, save=args.save, show=args.show)
            calc_ECDF_ranking(algs, areas, SR, dest_path, "feasibility", args.show, args.save)
            ECDF_group(algs, funs, dims, budgets, runs, optimality=False, save=args.save, show=args.show)

if __name__ == "__main__":
    main()