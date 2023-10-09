"""Module with functions to calculate statistics and rankings
"""
import numpy as np
import scipy.stats as scs
import csv

from utils.data_reader import get_results_from_file, get_properties
from utils.cec20_parameters import get_cec2020_constrained_optimum, get_cec2020_nonconstrained_optimum

properties = get_properties()

RESULT_LABEL = properties.get("RESULT_LABEL").data
ERROR_LABEL = properties.get("ERROR_LABEL").data
FES_CONSTRAINT_LABEL = properties.get("FES_CONSTRAINT_LABEL").data
FES_SUCCESS_LABEL = properties.get("FES_SUCCESS_LABEL").data
CONSTRAINT_VIOLATION_LABEL = properties.get("CONSTRAINT_VIOLATION_LABEL").data
MEAN_CONSTRAINT_VIOLATION_LABEL = properties.get("MEAN_CONSTRAINT_VIOLATION_LABEL").data
WILCOXON_DIR = properties.get("WILCOXON_DIR").data
AGGREGATED_RESULTS_DIR = properties.get("AGGREGATED_RESULTS_DIR").data
CEC_RANKING_DIR = properties.get("CEC_RANKING_DIR").data

def Wilcoxon(alg1, alg2, funs, dims, print_result=False, save=False):
    alg1_name, alg1_path = alg1
    alg2_name, alg2_path = alg2
    pvalues = 0
    total_better = 0
    total_ranks_positive = 0
    total_worse = 0
    total_ranks_negative = 0
    total_equal = 0
    res = []

    if print_result:
        print(alg1_name, "-", alg2_name)
        print("Problem  p-value  |    R+     R-  | Result")
        print("-" * 50)
    
    for n, fun in enumerate(funs):
        results = get_results_from_file(f'{alg1_path}/F{fun}-{dims[n]}D.csv')
        my_results = np.array([round(float(res[ERROR_LABEL]), 6) for res in results])
        results = get_results_from_file(f"{alg2_path}/F{fun}-{dims[n]}D.csv")
        original_results = np.array([round(float(res[ERROR_LABEL]), 6) for res in results])

        diff = my_results - original_results
        diff_abs = np.abs(my_results - original_results)
        wilc = scs.wilcoxon(my_results, original_results, zero_method="zsplit") if np.count_nonzero(np.where(diff_abs > 0)) < 8 else scs.wilcoxon(my_results, original_results)
        better = np.sum(diff < 0)
        worse = np.sum(diff > 0)
        equal = np.size(results) - better - worse
        ranks = assign_ranks(diff_abs)
        ranks_positive = np.sum(ranks[diff < 0])
        total_ranks_positive += ranks_positive
        ranks_negative = np.sum(ranks[diff > 0])
        total_ranks_negative += ranks_negative
        
        if wilc[1] >= 0.05:
            result = "="
            total_equal += 1
        elif max(better, equal, worse) == better:
            result = "+"
            total_better += 1
        else:
            result = "-"
            total_worse += 1
        
        pvalues += wilc[1]
        row = {"Problem": fun,
               "p-value": wilc[1],
               "R+": ranks_positive,
               "R-": ranks_negative,
               "Result": result,
                }
        res.append(row)

        if print_result:
            print(f"Problem {fun}: {wilc[1]:.4f} | {ranks_positive:6} {ranks_negative:6} | {result:^9}")
    
    if print_result:
        print("-" * 50)
        print(f"Total:      {(pvalues / len(funs)):.4} | {total_ranks_positive:6} {total_ranks_negative:6} | {total_better:2}/{total_equal:2}/{total_worse:2}")

    row = {"Problem": "Total",
            "p-value": pvalues / len(funs),
            "R+": total_ranks_positive,
            "R-": total_ranks_negative,
            "Result": f"{total_better}/{total_equal}/{total_worse}"
            }
    
    res.append(row)

    if save:
        path = f"{WILCOXON_DIR}/wilcoxon_{alg1_name}_{alg2_name}.csv"
        csvfile = open(path, 'w', newline='')
        fieldnames = ["Problem", "p-value", "R+", "R-", "Result"]
        spamwriter = csv.DictWriter(csvfile, delimiter=';', quotechar="|", quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        spamwriter.writeheader()
        spamwriter.writerows(res)
    
def calculate_stats(alg_name, results_path, funs, dims, with_constraints=False, print_results=False, save=False):
    results = []
    dest_path = AGGREGATED_RESULTS_DIR
    for fun, D in zip(funs, dims):
        
        outcome = np.array([])
        FES_feasible = np.array([])
        conv = np.array([])
        with open(f"{results_path}/F{fun}-{D}D.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')

            for row in reader:
                outcome = np.append(outcome, float(row[ERROR_LABEL]))
                if with_constraints:
                    FES_feasible = np.append(FES_feasible, int(float(row[FES_CONSTRAINT_LABEL])))
                    conv = np.append(conv, float(row[MEAN_CONSTRAINT_VIOLATION_LABEL]))

        if with_constraints:
            sorted_index = np.lexsort(np.column_stack((conv, outcome)).T[::-1])
            name = f"RC{fun} f"
            fr = (np.size(conv) - np.count_nonzero(conv)) / np.size(conv) * 100
            path = f"{dest_path}/{alg_name}_stats.csv"
        else:
            sorted_index = np.argsort(outcome)
            name = f"F{fun}"
            path = f"{dest_path}/{alg_name}_{dims[0]}D_stats.csv"
            

        mini = sorted_index[0]
        maxi = sorted_index[-1]
        sr = (np.size(outcome) - np.count_nonzero(outcome)) / np.size(outcome) * 100
        
        best = outcome[mini]
        worst = outcome[maxi]
        mean =  np.mean(outcome)
        median = np.median(outcome)
        std = np.std(outcome)

        row = {
            "Problem" : name,
            "Best" : best,
            "Worst" : worst,
            "Mean" : mean,
            "Median" : median,
            "Std." : std,
            "SR" : sr
        }

        if print_results:
            print("Problem", fun)
            print("Best:", best)
            print("Worst:", worst)
            print("Mean:", mean)
            print("Median:", median)
            print("Std.:", std)

        results.append(row)

        if with_constraints:
            best_conv = conv[mini]
            worst_conv = conv[maxi]
            mean_conv = np.mean(conv)
            median_conv = np.median(conv)
            std_conv = np.std(conv)
            row = {
                "Problem" : f"RC{fun} v",
                "Best" : best_conv,
                "Worst" : worst_conv,
                "Mean" : mean_conv,
                "Median" : median_conv,
                "Std." : std_conv,
                "SR" : fr,
            }

            if print_results:
                print("Best Constraint Violation:", best_conv)
                print("Worst Constraint Violation:", worst_conv)
                print("Mean Constraint Violation:", mean_conv)
                print("Median Constraint Violation:", median_conv)
                print("Std. Constraint Violation:", std_conv)

            results.append(row)
        if print_results:
            print()

    if save:
        csvfile = open(path, 'w', newline='')
        
        fieldnames = ["Problem", "Best", "Worst", "Mean", "Median", "Std.", "SR"]
        spamwriter = csv.DictWriter(csvfile, delimiter=';', quotechar="|", quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        spamwriter.writeheader()

        spamwriter.writerows(results)

def assign_ranks(arr, ascending=True):
    sorted_indices = np.argsort(arr)
    if not ascending:
        sorted_indices = sorted_indices[::-1]
        
    sorted_arr = arr[sorted_indices]
    
    _, unique_indices = np.unique(sorted_arr, return_index=True)
    
    ranks = np.empty(len(sorted_indices), dtype=float)
    
    for idx in unique_indices:
        same_val_indices = np.where(sorted_arr == sorted_arr[idx])[0]
        avg_rank = np.mean(same_val_indices) + 1
        ranks[same_val_indices] = avg_rank
    
    result_ranks = np.empty_like(arr, dtype=float)
    result_ranks[sorted_indices] = ranks
    
    return result_ranks

def calculate_cec2020nonconstrained_ranking(algs, print_results=True, save=False):
    optimum = get_cec2020_nonconstrained_optimum()
    funs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dims = [5, 10, 15, 20]

    bests = np.zeros((len(algs), len(funs), len(dims)))
    means = np.zeros((len(algs), len(funs), len(dims)))
    bests_max = np.zeros((len(funs), len(dims)))

    stats_path = AGGREGATED_RESULTS_DIR
    dest_path = CEC_RANKING_DIR

    for i, dim in enumerate(dims):
        
        cur_fun = 0
        for fun in funs:
            if dim == 5 and fun in [6, 7]:
                continue

            for n, alg_name in enumerate(algs):
                file_name = f"{stats_path}/{alg_name}_{dim}D_stats.csv"
                results = get_results_from_file(file_name)
                bests[n][cur_fun][i] = float(results[cur_fun]['Best']) + optimum[fun-1]
                means[n][cur_fun][i] = float(results[cur_fun]['Mean'])

            best_max = np.max(bests[:, cur_fun, i])
            bests_max[cur_fun][i] = best_max
            cur_fun += 1

    normalized_errors = np.zeros((len(algs), len(funs), len(dims)))
    ranks = np.zeros((len(algs), len(funs), len(dims)))

    for n, alg_name in enumerate(algs):
        for i, dim in enumerate(dims):
            cur_fun = 0
            for fun in funs:
                if dim == 5 and fun in [6, 7]:
                    continue

                #SCORE1
                best_max = bests_max[cur_fun][i]
                if best_max == optimum[fun-1]:
                    best_max = optimum[fun-1] + 1
                best_fun = bests[n][cur_fun][i]
                ne = (best_fun - optimum[fun-1]) / (best_max - optimum[fun-1])
                normalized_errors[n][cur_fun][i] = ne
                

                #SCORE2
                mean_f_d = means[:,cur_fun,i]
                rank = assign_ranks(mean_f_d)
                ranks[:,cur_fun,i] = rank

                cur_fun += 1

    normalized_errors
    SNE = np.zeros(len(algs))
    SR = np.zeros(len(algs))
    for n, alg_name in enumerate(algs):
        ne = normalized_errors[n]
        SNE[n] = 0.1 * np.sum(ne[:8,0]) + 0.2 * np.sum(ne[:,1]) + 0.3 * np.sum(ne[:,2]) + 0.4 * np.sum(ne[:,3])
        rank = ranks[n]
        SR[n] = 0.1 * np.sum(rank[:8,0]) + 0.2 * np.sum(rank[:,1]) + 0.3 * np.sum(rank[:,2]) + 0.4 * np.sum(rank[:,3])

    SNE_min = np.min(SNE)
    SR_min = np.min(SR)

    Scores1 = np.zeros(len(algs))
    Scores2 = np.zeros(len(algs))
    for n, alg_name in enumerate(algs):
        Scores1[n] = (1 - (SNE[n] - SNE_min) / SNE[n]) * 50
        Scores2[n] = (1 - (SR[n] - SR_min) / SR[n]) * 50

    PM = Scores1 + Scores2

    if print_results:
        print(f"{'Algorithm':15}   Score1     Score2    |    PM ")
        print("-" * 50)
        for i in range(len(algs)):
            print(f"{algs[i]:15}: {Scores1[i]:.6f}  {Scores2[i]:.6f}  | {PM[i]:.6f}")

    res = []
    for i in range(len(algs)):
        row = {"Algorithm" : algs[i],
               "Score1" : Scores1[i],
               "Score2" : Scores2[i],
               "PM" : PM[i]
               }
        res.append(row)

    if save:
        path = f"{dest_path}/cec20nonconstrained_ranking.csv"
        csvfile = open(path, 'w', newline='')
        fieldnames = ["Algorithm", "Score1", "Score2", "PM"]
        spamwriter = csv.DictWriter(csvfile, delimiter=';', quotechar="|", quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        spamwriter.writeheader()

        spamwriter.writerows(res)

def calculate_normalize_Af(bests, bests_conv, algs, funs):
    Af_best = np.zeros((len(algs), len(funs)))
    worsts_feasible_solutions = np.zeros(len(funs))
    worst_solution_index = np.argmax(bests[:], axis=0)
    for i in range(len(funs)):
        worst_feasible_solution = bests[worst_solution_index[i],i] if bests_conv[worst_solution_index[i], i] == 0 else 0
        worsts_feasible_solutions[i] = worst_feasible_solution

    for n in range(len(algs)):
        for fun in funs:
            if bests_conv[n][fun-1] > 0:
                Af_best[n][fun-1] = worsts_feasible_solutions[fun-1] + bests_conv[n][fun-1]
            else:
                Af_best[n][fun-1] = bests[n][fun-1]

    Af_mins = np.min(Af_best[:], axis=0)
    Af_maxs = np.max(Af_best[:], axis=0)
    
    normalized_Af_best = (Af_best - Af_mins) / (Af_maxs - Af_mins)
    normalized_Af_best[np.isnan(normalized_Af_best)] = 0

    return normalized_Af_best

def calculate_cec2020constrained_ranking(algs, print_results=True, save=False):
    optimum = get_cec2020_constrained_optimum()
    funs = list(range(1, 21))
    dims = np.array([9, 11, 7, 6, 9, 38, 48, 2, 3, 3, 7, 7, 5, 10, 7, 14, 3, 4, 4, 2])
    weights = np.zeros(len(funs))

    weights[dims <= 10] = 0.008
    weights[(dims > 10) & (dims <= 30)] = 0.016
    weights[(dims > 30) & (dims <= 50)] = 0.024

    bests = np.zeros((len(algs), len(funs)))
    bests_conv = np.zeros((len(algs), len(funs)))
    means = np.zeros((len(algs), len(funs)))
    means_conv = np.zeros((len(algs), len(funs)))
    medians = np.zeros((len(algs), len(funs)))
    medians_conv = np.zeros((len(algs), len(funs)))

    stats_path = AGGREGATED_RESULTS_DIR
    dest_path = CEC_RANKING_DIR
    
    for n, alg_name in enumerate(algs):
        file_name = f"{stats_path}/{alg_name}_stats.csv"
        results = get_results_from_file(file_name)
        for fun in funs:
            
            bests[n][fun-1] = float(results[(fun-1) * 2]["Best"]) + optimum[fun-1]
            bests_conv[n][fun-1] = float(results[(fun-1) * 2 + 1]["Best"])
            means[n][fun-1] = float(results[(fun-1) * 2]["Mean"])
            means_conv[n][fun-1] = float(results[(fun-1) * 2 + 1]["Mean"])
            medians[n][fun-1] = float(results[(fun-1) * 2]["Median"])
            medians_conv[n][fun-1] = float(results[(fun-1) * 2 + 1]["Median"])

    norm_Af_best = calculate_normalize_Af(bests, bests_conv, algs, funs)
    norm_Af_mean = calculate_normalize_Af(means, means_conv, algs, funs,)
    norm_Af_median = calculate_normalize_Af(medians, medians_conv, algs, funs)
    
    
    Scores1 = 0.5 * np.dot(norm_Af_best, weights)
    Scores2 = 0.3 * np.dot(norm_Af_mean, weights)
    Scores3 = 0.2 * np.dot(norm_Af_median, weights)

    PM = Scores1 + Scores2 + Scores3

    if print_results:
        print(f"{'Algorithm':15}   Score1    Score2    Score3  |   PM ")
        print("-" * 56)
        for i in range(len(algs)):
            print(f"{algs[i]:15}: {Scores1[i]:.6f}  {Scores2[i]:.6f}  {Scores3[i]:.6f} | {PM[i]:.6f}")

    res = []
    for i in range(len(algs)):
        row = {"Algorithm" : algs[i],
               "Score1" : Scores1[i],
               "Score2" : Scores2[i],
               "Score3" : Scores3[i],
               "PM" : PM[i]
               }
        res.append(row)


    if save:
        path = f"{dest_path}/cec20constrained_ranking.csv"
        csvfile = open(path, 'w', newline='')
        fieldnames = ["Algorithm", "Score1", "Score2", "Score3", "PM"]
        spamwriter = csv.DictWriter(csvfile, delimiter=';', quotechar="|", quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        spamwriter.writeheader()

        spamwriter.writerows(res)

def print_ranking(algs, ranking, name):
    print(name)
    print(f"{'Problem':11}", end="")
    for i in range(ranking.shape[1]):
        print(f"{i+1:5} |", end="")
    print(" sum")
    for i, (alg, _) in enumerate(algs):
        print(f"{alg:11}", end=" ")
        for j in range(ranking.shape[1]):
            print(f"{ranking[i][j]:4} | ", end="")
        print(np.sum(ranking[i,:]))

def save_ranking(algs, ranking, path):
    res = []
    for i in range(len(algs)):
        row = {"Algorithm" : algs[i][0]}
        
        for j in range(ranking.shape[1]):
            row[str(j+1)] = ranking[i][j]
        
        row["Total"] = np.sum(ranking[i,:])
        res.append(row)

    csvfile1 = open(path, 'w', newline='')
    fieldnames = ["Algorithm"]
    for j in range(ranking.shape[1]):
        fieldnames.append(str(j+1))
    fieldnames.append("Total")
    spamwriter = csv.DictWriter(csvfile1, delimiter=';', quotechar="|", quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
    spamwriter.writeheader()
    spamwriter.writerows(res)
    
def calc_ECDF_ranking(algs, areas, SR, path, name, print_results=True, save=False):
    ranks_area = np.zeros(SR.shape)
    ranks_SR = np.zeros(SR.shape)
    for i in range(SR.shape[1]):
        fun_areas = areas[:, i]
        rank_area = assign_ranks(fun_areas, False)
        ranks_area[:, i] = rank_area
        fun_SR = SR[:, i]
        rank_SR = assign_ranks(fun_SR, False)
        ranks_SR[:, i] = rank_SR

    if print_results:
        print_ranking(algs, ranks_area, "Area under curve")
        print_ranking(algs, ranks_SR, "SR")

    if save:
        save_ranking(algs, ranks_area, f"{path}/ranking_area_{name}.csv")
        save_ranking(algs, ranks_SR, f"{path}/ranking_SR_{name}.csv")

    return ranks_area, ranks_SR
