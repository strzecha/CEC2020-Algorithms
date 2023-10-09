"""Script to perform Wilcoxon test
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.cec20_parameters import initialize_cec2020_constrained_parameters
from utils.stats import Wilcoxon
from utils.data_reader import get_properties


def main():
    properties = get_properties()

    BENCHMARKS_CONSTRAINED = properties.get("BENCHMARKS_CONSTRAINED").data.split(",")
    BENCHMARKS_NONCONSTRAINED = properties.get("BENCHMARKS_NONCONSTRAINED").data.split(",")
    BENCHMARKS_NAMES = BENCHMARKS_CONSTRAINED + BENCHMARKS_NONCONSTRAINED
    ALGORITHMS_NAMES = properties.get("ALGORITHMS").data.split(",")
    RESULTS_DIR = properties.get("RESULTS_DIR").data


    parser = argparse.ArgumentParser()

    parser.add_argument('benchmark', choices=BENCHMARKS_NAMES)
    parser.add_argument('algorithm1', choices=ALGORITHMS_NAMES, type=str)
    parser.add_argument('algorithm2', choices=ALGORITHMS_NAMES, type=str)
    parser.add_argument('-f', '--functions', type=int, nargs='*')
    parser.add_argument('-l', '--all', action='store_true')
    parser.add_argument('-d', '--dim', type=int)
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-s', '--show', action="store_true")
    parser.add_argument('-v', '--save', action="store_true")

    args = parser.parse_args()

    if args.show or args.save:
        if args.path:
            path = args.path
        else:
            path = RESULTS_DIR

        algs = []

        for alg_name in [args.algorithm1, args.algorithm2]:
            alg_path = f"{path}/{alg_name}/{args.benchmark}"
            alg = [alg_name, alg_path]
            algs.append(alg)


        if args.benchmark == BENCHMARKS_NONCONSTRAINED[0]:
            if not args.dim:
                raise Exception("You must specify dimensionality for CEC20nonconstr benchmark")
            if args.all:
                if args.dim != 5:
                    funs = list(range(1, 11))
                else:
                    funs = [1, 2, 3, 4, 5, 8, 9, 10]
            else:
                funs = args.functions

            dims = [args.dim] * len(funs)
            
        if args.benchmark == BENCHMARKS_CONSTRAINED[0]:
            dims = []
            if args.all:
                funs = list(range(1, 21))
            else:
                funs = args.functions

            for fun in funs:
                _, _, _, D = initialize_cec2020_constrained_parameters(fun)
                dims.append(D)

        Wilcoxon(*algs, funs, dims, args.show, args.save)

if __name__ == "__main__":
    main()