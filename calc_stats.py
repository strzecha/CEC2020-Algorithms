"""Script to calculate statistics
"""
import argparse

from utils.cec20_parameters import initialize_cec2020_constrained_parameters
from utils.stats import calculate_stats
from utils.data_reader import get_properties

def main():
    properties = get_properties()

    BENCHMARKS_CONSTRAINED = properties.get("BENCHMARKS_CONSTRAINED").data.split(",")
    BENCHMARKS_NONCONSTRAINED = properties.get("BENCHMARKS_NONCONSTRAINED").data.split(",")
    BENCHMARKS_NAMES = BENCHMARKS_CONSTRAINED + BENCHMARKS_NONCONSTRAINED
    ALGORITHMS_NAMES = properties.get("ALGORITHMS").data.split(",")

    parser = argparse.ArgumentParser()

    parser.add_argument('benchmark', choices=BENCHMARKS_NAMES)
    parser.add_argument('algorithm', choices=ALGORITHMS_NAMES, type=str)
    parser.add_argument('-f', '--functions', type=int, nargs='*')
    parser.add_argument('-l', '--all', action='store_true')
    parser.add_argument('-d', '--dim', type=int)
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-s', '--show', action="store_true")
    parser.add_argument('-v', '--save', action="store_true")

    args = parser.parse_args()

    if args.show or args.save:
        if args.path:
            root = args.path
        else:
            root = properties.get("RESULTS_DIR").data

        path = f"{root}/{args.algorithm}/{args.benchmark}"


        with_constraints = False
        if args.benchmark in BENCHMARKS_CONSTRAINED:
            with_constraints = True

        if args.benchmark == BENCHMARKS_NONCONSTRAINED[0]:
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
            
        if args.benchmark == BENCHMARKS_CONSTRAINED[0]:
            if args.all:
                funs = list(range(1, 21))
            else:
                funs = args.functions
            dims = []
            for fun in funs:
                _, _, _, D = initialize_cec2020_constrained_parameters(fun)
                dims.append(D)

        calculate_stats(args.algorithm, path, funs, dims, with_constraints, args.show, args.save)


if __name__ == "__main__":
    main()