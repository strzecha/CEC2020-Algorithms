import numpy as np
from scipy.stats import norm, levy_stable

print("random imported")

file_rand = open("random_generator/random_numbers.txt")
file_randint = open("random_generator/random_numbers.txt")
file_normal_r = open("random_generator/random_numbers.txt")
file_randn = open("random_generator/randn.txt")

file_cauchy = open("random_generator/pd_cauchy.txt")
file_normal = open("random_generator/pd_normal.txt")
file_levy0 = open("random_generator/pd_levy0.txt")

def my_permutation(n):
    random_indices = np.random.rand(n)
    sorted_indices = np.argsort(random_indices)
    return sorted_indices

def random_normal_custom(mean=0.0, std=1.0, ):
    size = mean.shape
    output = np.zeros(size)
    
    for index in np.ndindex(size):
        u1 = 1.0 - get_num_normal_r()
        u2 = 1.0 - get_num_normal_r()
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
        output[index] = mean[index] + std * z
    
    return output
    
def get_normal():
    global file_normal
    num = file_normal.readline()
    if num == "":
        file_normal.close()
        file_normal = open("random_generator/pd_normal.txt")
        num = file_normal.readline()
    return float(num)

def get_cauchy():
    global file_cauchy
    num = file_cauchy.readline()
    if num == "":
        file_cauchy.close()
        file_cauchy = open("random_generator/pd_cauchy.txt")
        num = file_cauchy.readline()
    return float(num)

def get_levy0():
    global file_levy0
    num = file_levy0.readline()
    if num == "":
        file_levy0.close()
        file_levy0 = open("random_generator/pd_levy0.txt")
        num = file_levy0.readline()
    return float(num)

def get_num_rand():
    global file_rand
    num = file_rand.readline()
    if num == "":
        file_rand.close()
        file_rand = open("random_generator/random_numbers.txt")
        num = file_rand.readline()
    return float(num)

def get_num_randn():
    global file_randn
    num = file_randn.readline()
    if num == "":
        file_randn.close()
        file_randn = open("random_generator/randn.txt")
        num = file_randn.readline()
    return float(num)

def get_num_normal_r():
    global file_normal_r
    num = file_normal_r.readline()
    if num == "":
        file_normal_r.close()
        file_normal_r = open("random_generator/random_numbers.txt")
        num = file_normal_r.readline()
    return float(num)

def get_num_randint():
    global file_randint
    num = file_randint.readline()
    
    if num == "":
        file_randint.close()
        file_randint = open("random_generator/random_numbers.txt")
        num = file_randint.readline()
    return float(num)

def custom_rand(a=1, b=None):
    if a == 1 and b is None:
        return float(get_num_rand())
    if b is None:
        return np.array([float(get_num_rand()) for i in range(a)])
    return np.array([np.array([float(get_num_rand()) for i in range(b)]) for j in range(a)])

def custom_randn(a=1, b=None):
    if a == 1 and b is None:
        return get_num_randn()
    if b is None:
        return np.array([get_num_randn() for i in range(a)])
    return np.array([np.array([get_num_randn() for i in range(b)]) for j in range(a)])

def randint_custom(low, high, size=None):
    if size is None:
        # Jeśli argument `size` nie jest podany, zwróć pojedynczą liczbę całkowitą
        num = get_num_randint()
        return low + int(num * (high - low))
    else:
        # Jeśli podany jest argument `size`, zwróć tablicę losowych liczb całkowitych
        size = np.array(size)  # Konwertuj `size` na tablicę numpy
        random_integers = np.zeros(size, dtype=int)  # Inicjalizuj tablicę wynikową

        # Generowanie losowych liczb całkowitych dla każdego elementu tablicy
        for index, element in np.ndenumerate(random_integers):
            num = get_num_randint()
            random_integers[index] = low + int(num * (high - low))

        return random_integers

def custom_levy(size=None):
    return np.array([np.array([get_levy() for i in range(size[1])]) for j in range(size[0])])

def custom_levy0(size=None):
    return np.array([np.array([get_levy0() for i in range(size[1])]) for j in range(size[0])])

def my_round(arr, decimals=0):
    multiplier = 10 ** decimals
    rounded_values = np.floor(arr * multiplier + 0.5) / multiplier
    return rounded_values.astype(int) if decimals == 0 else rounded_values

np.round = my_round

def restart_generator():
    global file_rand
    global file_randint
    global file_normal_r
    global file_randn

    global file_cauchy
    global file_normal
    global file_levy0


    file_rand = open("random_generator/random_numbers.txt")
    file_randint = open("random_generator/random_numbers.txt")
    file_normal_r = open("random_generator/random_numbers.txt")
    file_randn = open("random_generator/randn.txt")

    file_cauchy = open("random_generator/pd_cauchy.txt")
    file_normal = open("random_generator/pd_normal.txt")
    file_levy0 = open("random_generator/pd_levy0.txt")

def custom_random_choice(arr, size=1):
    random_indices = np.floor(np.random.rand(size) * len(arr)).astype(int)

    return arr[random_indices]

np.random.choice = custom_random_choice
np.random.rand = custom_rand
np.random.randint = randint_custom
np.random.normal = random_normal_custom
np.random.randn = custom_randn
np.random.permutation = my_permutation