import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

from utils.data_reader import get_properties

properties = get_properties()


PLOT_DIR = properties.get("PLOT_DIR").data
RESULT_LABEL = properties.get("RESULT_LABEL").data
ERROR_LABEL = properties.get("ERROR_LABEL").data
FES_CONSTRAINT_LABEL = properties.get("FES_CONSTRAINT_LABEL").data
FES_SUCCESS_LABEL = properties.get("FES_SUCCESS_LABEL").data
CONSTRAINT_VIOLATION_LABEL = properties.get("CONSTRAINT_VIOLATION_LABEL").data
MEAN_CONSTRAINT_VIOLATION_LABEL = properties.get("MEAN_CONSTRAINT_VIOLATION_LABEL").data

def save_or_show(save, show, name):
    if save:
        date = datetime.datetime.now()
        date_print = f"{date.day}-{date.month}-{date.hour}.{date.minute}.{date.second}"
        path = f"{PLOT_DIR}/{name}-{date_print}.png"
        plt.tight_layout()
        plt.savefig(path)

    if show:
        plt.tight_layout()
        plt.show()

def ECDF_xaxis_formula(FES, MAX_FES, D):
    return FES / MAX_FES

def get_labels(optimality):
    if optimality:
        name = "optimality"
        criterion = 1
    else:
        name = "feasibility"
        criterion = 2

    return name, criterion
    
def ECDF_group(algs, funs, dims, FES_budgets, n_runs, optimality=True, show=False, save=False):
    name, criterion = get_labels(optimality)
    markers = ('*', 'h', 'H', 'D', 'd', 'P', 'X')
    _, ax = plt.subplots(1, 1, figsize=(7, 6))
    for i, (alg_name, root) in enumerate(algs):
        style = f"-{markers[i]}"
        Xs = []
        Ys = []
        for n, fun in enumerate(funs):
            runs = []
            for run in range(n_runs):
                FES_budget = FES_budgets[n]
                filename = f"{root}/F{fun}-{dims[n]}D-runs/{run+1}.npy"
                runs.append(np.load(filename))

            X, Y = get_X_Y(runs, FES_budget, criterion)

            Xs.append(X)
            Ys.append(Y)
        
        Y = np.mean(np.array(Ys), axis=0)
        area = np.round(np.trapz(Y, X) / ECDF_xaxis_formula(FES_budget, FES_budget, dims[n]), 3)
        label=f"{alg_name}, S={area}"
        draw_ECDF(X, Y, ax, f"", label, style)

    save_or_show(save, show, name)

def ECDF_per_function(algs, funs, dims, FES_budgets, n_runs, row, col, optimality=True, show=False, save=False):
    width = col * 5.25
    height = row * 4
    _, ax = plt.subplots(row, col, figsize=(width, height))
    if row == 1 and col != 1:
        ax = ax.reshape(1, col)
    name, criterion = get_labels(optimality)
    markers = ('*', 'h', 'H', 'D', 'd', 'P', 'X')
    areas = np.zeros((len(algs), len(funs)))
    SR = np.zeros((len(algs), len(funs)))
    for i, (alg_name, root) in enumerate(algs):
        style = f"-{markers[i]}"
        for n, fun in enumerate(funs):
            runs = []
            for run in range(n_runs):
                FES_budget = FES_budgets[n]
                filename = f"{root}/F{fun}-{dims[n]}D-runs/{run+1}.npy"
                runs.append(np.load(filename))

            X, Y = get_X_Y(runs, FES_budget, criterion)
            x = n // col
            y = n % col
            area = np.round(np.trapz(Y, X) / ECDF_xaxis_formula(FES_budget, FES_budget, dims[n]), 3)

            areas[i][n] = area
            SR[i][n] = Y[-1]

            label=f"{alg_name}, S={area}"
            title = f"F{fun}-{dims[n]}D"

            if col == row == 1:
                draw_ECDF(X, Y, ax, title, label, style)
            else:
                draw_ECDF(X, Y, ax[x, y], title, label, style)

    save_or_show(save, show, name)

    return areas, SR

def get_X_Y(results, max_fes, choice):
    thresholds = 10 ** np.linspace(np.log10(1e3), np.log10(1e-8), 51)
    bests = np.zeros(len(results))
    Xs = np.linspace(0, 1, 1001)

    for j in range(len(results)):
        bests[j] = float(results[j][0][choice])

    X = []
    Y = [0]
    i = 0

    indexes = np.zeros(len(results)).astype(int)
    feasible = False
    for i in range(len(Xs) - 1):
        sum = 0
        for j in range(len(results)):
            val_show = bests[j]
            if results[j].shape[0] > indexes[j]:
                while indexes[j] < results[j].shape[0] and (float(results[j][indexes[j]][0]) / max_fes) <= Xs[i]:
                    val = float(results[j][indexes[j]][choice])
                    val_show = min(val, bests[j])

                    if results[j].shape[1] > 2:
                        svc = float(results[j][indexes[j]][2])
                    else:
                        svc = 0
                    if svc == 0:
                        feasible = True
                    indexes[j] += 1

            bests[j] = val_show
            sum += (np.sum(val_show < thresholds) / len(thresholds))
        X.append(Xs[i])
        if (choice == 1 and feasible) or choice == 2: 
            Y.append(sum / len(results)) 
        else:
            Y.append(0)

    X.append(Xs[-1])

    return X, Y
    
def draw_ECDF(X, Y, ax, title, label, style):
    first_non_zero = np.argmax(np.array(Y) > 0)
    ax.plot(X, Y, style, label=label, markevery=50)
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.set_xlim([X[first_non_zero], 1])
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_xlabel('FES / MAX_FES')
    ax.set_ylabel('Proporcja osiągniętych progów')



