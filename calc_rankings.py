"""Script to calculate CEC2020 rankings
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.stats import calculate_cec2020constrained_ranking, calculate_cec2020nonconstrained_ranking
from utils.data_reader import get_properties


def main():
    properties = get_properties()

    BENCHMARKS_CONSTRAINED = properties.get("BENCHMARKS_CONSTRAINED").data.split(",")
    BENCHMARKS_NONCONSTRAINED = properties.get("BENCHMARKS_NONCONSTRAINED").data.split(",")
    BENCHMARKS_NAMES = BENCHMARKS_CONSTRAINED + BENCHMARKS_NONCONSTRAINED
    ALGORITHMS_NAMES = properties.get("ALGORITHMS").data.split(",")

    parser = argparse.ArgumentParser()

    parser.add_argument('benchmark', choices=BENCHMARKS_NAMES)
    parser.add_argument('-a', '--algorithms', choices=ALGORITHMS_NAMES, type=str, nargs='+', required=True)
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('-v', "--save", action="store_true")

    args = parser.parse_args()

    if args.show or args.save:
        if args.benchmark == BENCHMARKS_CONSTRAINED[0]:
            calculate_cec2020constrained_ranking(args.algorithms, args.show, args.save)

        if args.benchmark == BENCHMARKS_NONCONSTRAINED[0]:
            calculate_cec2020nonconstrained_ranking(args.algorithms, args.show, args.save)

if __name__ == "__main__":
    main()