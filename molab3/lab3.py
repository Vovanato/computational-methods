import csv
import math
import matplotlib.pyplot as plt

# 1. Вхідні дані 
def load_data(filename):
    x, y = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропускаємо заголовок Month, Temp
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y

# 2. Функції МНК 
def form_matrix(x, m):
    # Створюємо нульову матрицю (m+1) x (m+1)
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(xi**(i + j) for xi in x)
    return A

def form_vector(x, y, m):
    # Створюємо нульовий вектор (m+1)
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * (x[k]**i) for k in range(len(x)))
    return b

def gauss_solve(A, b):
    n = len(b)
    # Копіюємо матриці, щоб не змінювати оригінали
    A = [row[:] for row in A]
    b = b[:]
    
    # Прямий хід з вибором головного елемента по стовпцю
    for k in range(n - 1):
        # Знаходимо рядок з найбільшим за модулем елементом
        max_row = k + max(range(n - k), key=lambda i: abs(A[k + i][k]))
        
        # Міняємо рядки місцями
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]
        
        for i in range(k + 1, n):
            if A[k][k] == 0: continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]
            
    # Зворотній хід
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x_sol[j] for j in range(i + 1, n))
        if A[i][i] != 0:
            x_sol[i] = (b[i] - s) / A[i][i]
    return x_sol

def polynomial(x_vals, coef):
    y_poly = []
    for xi in x_vals:
        val = sum(coef[i] * (xi**i) for i in range(len(coef)))
        y_poly.append(val)
    return y_poly

def variance(y_true, y_approx):
    n = len(y_true)
    # Формула дисперсії
    return sum((y_true[i] - y_approx[i])**2 for i in range(n)) / n

def calculate_error(y_true, y_approx):
    # Похибка e(x) = |f(x) - phi(x)|
    return [abs(y_true[i] - y_approx[i]) for i in range(len(y_true))]

# Лінійна інтерполяція для знаходження f(x) між вузлами (для дрібного кроку h1)
def interpolate_true_y(x_true, y_true, x_target):
    for i in range(len(x_true) - 1):
        if x_true[i] <= x_target <= x_true[i+1]:
            x0, y0 = x_true[i], y_true[i]
            x1, y1 = x_true[i+1], y_true[i+1]
            return y0 + (y1 - y0) * ((x_target - x0) / (x1 - x0))
    return y_true[-1]

#Головний блок програми 
def main():
    x, y = load_data('data.csv')
    n = len(x)
    
    # 3. Вибір оптимального ступеня полінома (обмежено до m=4 як у звіті)
    max_degree = 4
    variances = []
    best_coefs = {}
    
    print(f"Дисперсії для різних ступенів (m=1..{max_degree}):")
    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        variances.append(var)
        best_coefs[m] = coef
        print(f" Ступінь {m}: дисперсія = {var:.4f}")
        
    optimal_m = variances.index(min(variances)) + 1
    optimal_coef = best_coefs[optimal_m]
    print(f"\nОптимальний ступінь полінома: {optimal_m}")
    
    # Вивід коефіцієнтів у науковому форматі, як у звіті
    coef_str = " ".join([f"{c:.8e}" for c in optimal_coef])
    print(f"Коефіцієнти полінома: [{coef_str}]")
    
    # 4. Прогноз на наступні 3 місяці
    x_future = [25, 26, 27]
    y_future = polynomial(x_future, optimal_coef)
    
    future_str = " ".join([f"{f:.8f}" for f in y_future])
    print(f"Прогноз на наступні 3 місяці (25, 26, 27): [{future_str}]")
    
    #  Побудова графіків 
    
    # Графік 1: Залежність дисперсії від ступеня (Окреме вікно)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_degree + 1), variances, marker='o', markersize=8)
    plt.title('Залежність дисперсії від ступеня')
    plt.xlabel('Ступінь m')
    plt.ylabel('Дисперсія')
    plt.xticks(range(1, max_degree + 1))
    plt.grid(True)
    
    # Графік 2: Апроксимація та похибка (Одне вікно, 2 рядки)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Верхній графік: Апроксимація
    ax1.scatter(x, y, color='blue', label='Фактичні температури', zorder=5)
    
    x_smooth = [x[0] + i*(x_future[-1]-x[0])/100 for i in range(101)]
    y_smooth = polynomial(x_smooth, optimal_coef)
    ax1.plot(x_smooth, y_smooth, label=f'Апроксимація (m={optimal_m})', color='darkgreen')
    
    ax1.scatter(x_future, y_future, color='crimson', marker='x', s=100, label='Прогноз (25-27)', zorder=5)
    ax1.plot([x[-1]] + x_future, [polynomial([x[-1]], optimal_coef)[0]] + y_future, color='crimson', linestyle='--')
    
    ax1.set_title('Апроксимація МНК та прогноз температур')
    ax1.set_xlabel('Місяці')
    ax1.set_ylabel('Температура')
    ax1.legend()
    ax1.grid(True)
    
    # Нижня гістограма: Похибка (фактичне - апроксимація)
    y_optimal_approx = polynomial(x, optimal_coef)
    error_y = [y[i] - y_optimal_approx[i] for i in range(n)]
    
    ax2.bar(x, error_y, color='orange', edgecolor='lightgray', label='Похибка (фактичне - апроксимація)')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title('Похибка апроксимації')
    ax2.set_xlabel('Місяці')
    ax2.set_ylabel('Величина похибки')
    ax2.legend()
    ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()