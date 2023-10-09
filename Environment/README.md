# CEC2020 Algorithms

Environment to comparision of evolutionary algorithms based on CEC benchmarks.

Constaints Python implementations of the best algorithms from CEC2020:

### CEC2020 - Special Session & Competitions on Real-World Constrained Optimization
- COLSHADE
- EnMODE
- esCMAgES

### CEC2020 - Competition on Single Objective Bound Constrained Numerical Optimization
- AGSK
- IMODE
- j2020

## Run
There are some operations you can do in this environment:

### Optimization
To run optimization of functions type in terminal:
```
python optimize.py BENCHMARK ALGORITH FUNCTION DIMENSIONALITY RUNS [-f BUDGET] [-p PATH]
```

### Aggregate results
To aggreagate results of optimization and calculate statistics type in terminal:
```
python calc_stats.py BENCHMARK ALGORITH [-f [FUNCTIONS ...]] [-l] [-d DIM] [-p PATH] [-s] [-v]
```

### ECDF curves
To draw ECDF curves based on results of optimization and calculate rankings type in terminal:
```
python ecdf.py BENCHMARK -a [ALGORITHMS ...] [-f [FUNCTIONS ...]] [-l] [-d DIM] [-c] [-p PATH] [-b BUDGET] [-s] [-v]
```

### Wilocoxon test
To perform Wilcoxon test between two algorithms type in terminal:
```
python wilcoxon.py ALGORITHM1 ALGORITHM2 [-f [FUNCTIONS ...]] [-l] [-d DIM] [-p PATH] [-s] [-v]
```

### CEC2020 rankings
To calculate CEC2020 rankings type in terminal:
```
python calc_rankings.py BENCHMARK -a [ALGORITHMS ...] [-s] [-v]
```
