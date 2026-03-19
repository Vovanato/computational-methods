import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt
import numpy as np

def divided_differences(x, y):
    n = len(y)
    coef = [list(y)]
    
    for j in range(1, n):
        row = []
        for i in range(n - j):
            res = (coef[j-1][i+1] - coef[j-1][i]) / (x[i+j] - x[i])
            row.append(res)
        coef.append(row)
    
    return [c[0] for c in coef]
def newton_interpolation(x_nodes, y_nodes, x_target):
    coef = divided_differences(x_nodes, y_nodes)
    n = len(coef)
    result = coef[0]
    product = 1.0
    
    for i in range(1, n):
        product *= (x_target - x_nodes[i-1]) 
        result += coef[i] * product
        
    return result
def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)

try:
    x_data, y_data = read_data('data.csv')
    
    # Прогноз для 120,000 
    target_n = 120000
    prediction = newton_interpolation(x_data, y_data, target_n)
    print(f"Прогнозований час для n={target_n}: {prediction:.2f} сек")

    # 3. Візуалізація []
    x_range = np.linspace(min(x_data), max(x_data), 500)
    y_interp = [newton_interpolation(x_data, y_data, x) for x in x_range]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Експериментальні дані')
    plt.plot(x_range, y_interp, label='Інтерполяція Ньютона')
    plt.scatter([target_n], [prediction], color='green', marker='x', s=100, label='Прогноз')
    
    plt.title('Прогноз часу тренування моделі ML (Варіант 3)')
    plt.xlabel('Розмір датасету (n)')
    plt.ylabel('Час (сек)')
    plt.legend()
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print("Помилка: Спершу створи файл data.csv з даними твого варіанту!")

#1. Метод Ньютона
def divided_differences(x, y):
    n = len(y)
    coef = [list(y)]
    for j in range(1, n):
        row = []
        for i in range(n - j):
            res = (coef[j-1][i+1] - coef[j-1][i]) / (x[i+j] - x[i])
            row.append(res)
        coef.append(row)
    return [c[0] for c in coef]

def newton_interpolation(x_nodes, y_nodes, x_target):
    coef = divided_differences(x_nodes, y_nodes)
    n = len(coef)
    result = coef[0]
    product = 1.0
    for i in range(1, n):
        product *= (x_target - x_nodes[i-1]) 
        result += coef[i] * product
    return result

#  2. Метод Факторіальних многочленів 
def finite_differences(y):
    n = len(y)
    coef = [list(y)]
    for j in range(1, n):
        row = []
        for i in range(n - j):
            row.append(coef[j-1][i+1] - coef[j-1][i])
        coef.append(row)
    return [c[0] for c in coef]

def factorial_interpolation(x_nodes, y_nodes, x_target):
    # Крок h (працює коректно лише для рівновіддалених вузлів)
    h = x_nodes[1] - x_nodes[0]
    t = (x_target - x_nodes[0]) / h
    
    coef = finite_differences(y_nodes)
    n = len(coef)
    
    result = coef[0]
    t_term = 1.0
    
    for i in range(1, n):
        t_term *= (t - (i - 1))
        result += (coef[i] * t_term) / math.factorial(i)
        
    return result

#  Зчитування даних 
def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)

#  Головний блок 
try:
    x_data, y_data = read_data('data.csv')
    
    # Створюємо "ідеальну" неперервну функцію на базі 5 точок для тестування
    true_func = interp1d(x_data, y_data, kind='cubic', fill_value="extrapolate")
    
    # Густа сітка для малювання графіків
    x_dense = np.linspace(10000, 160000, 150)
    y_true = true_func(x_dense)

    # Рівновіддалені вузли для порівняння Ньютона і Факторіального (бо Факторіальний вимагає сталого кроку)
    x_nodes_5 = np.linspace(10000, 160000, 5)
    y_nodes_5 = true_func(x_nodes_5)

    y_newton_5 = [newton_interpolation(x_nodes_5, y_nodes_5, x) for x in x_dense]
    y_fact_5 = [factorial_interpolation(x_nodes_5, y_nodes_5, x) for x in x_dense]
    error_diff = np.abs(np.array(y_fact_5) - np.array(y_newton_5))
    
    plt.figure(figsize=(10, 4))
    plt.plot(x_dense, error_diff, color='darkmagenta', label='Error |Factorial - Newton|')
    plt.title('Absolute error')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    
    nodes_list = [5, 10, 20]
    
    # Кольори ліній, щоб співпадало зі скріншотом
    colors_newton = ['#1f77b4', '#ff7f0e', '#00b300'] # синій, помаранчевий, яскраво-зелений
    colors_fact = ['#d62728', '#9467bd', '#8c564b']   # червоний, фіолетовий, коричневий

    for i, n_nodes in enumerate(nodes_list):
        # Генеруємо n рівномірних вузлів
        x_nodes = np.linspace(10000, 160000, n_nodes)
        y_nodes = true_func(x_nodes)
        
        # Обчислюємо інтерполяції
        y_interp_newton = np.array([newton_interpolation(x_nodes, y_nodes, x) for x in x_dense])
        y_interp_fact = np.array([factorial_interpolation(x_nodes, y_nodes, x) for x in x_dense])
        
        # Похибки відносно "ідеальної" функції
        err_newton = np.abs(y_true - y_interp_newton)
        err_fact = np.abs(y_true - y_interp_fact)
        
        # Графіки Ньютона (суцільні лінії)
        plt.plot(x_dense, err_newton, 
                 color=colors_newton[i], 
                 linestyle='-', 
                 label=f'|Похибка| {n_nodes} вузлів (Ньютон)')
        
        # Графіки Факторіального (пунктирні лінії)
        plt.plot(x_dense, err_fact, 
                 color=colors_fact[i], 
                 linestyle='--', 
                 label=f'|Похибка| {n_nodes} вузлів (Факт.)')

    # Оформлення другого графіка
    plt.xlabel('n')
    plt.ylabel('Абсолютна похибка')
    
    # Легенда зверху у 3 колонки без рамки
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    
    # Прибираємо верхню і праву рамку графіка для візуальної відповідності
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Помилка: Спершу створи файл data.csv з даними твого варіанту!")