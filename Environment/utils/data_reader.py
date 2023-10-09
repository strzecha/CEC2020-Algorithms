import numpy as np
import csv
from jproperties import Properties


def get_properties(file_name="properties"):
    configs = Properties()
    with open(file_name, 'rb') as config_file:
        configs.load(config_file)

    return configs

def get_results_from_file(filename):
    results = np.array([])
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            results = np.append(results, row)

    return results
