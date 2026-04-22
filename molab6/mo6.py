import numpy as np

def generate_data(n=100, x_true_val=2.5):
    # 1. Генеруємо матрицю А та вектор B
    A = np.random.uniform(1, 100, (n, n))
    x_true = np.full(n, x_true_val)
    b = A @ x_true
    
    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', b)
    return n

def lu_decomposition(A, n):
    L = np.zeros((n, n))
    U = np.eye(n) # Діагональні елементи U = 1

    for k in range(n):
        # Обчислення стовпців L
        for i in range(k, n):
            L[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(k))
        
        # Обчислення рядків U
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(k))) / L[k, k]
            
    return L, U

def solve_lu(L, U, b, n):
    # LZ = B (Пряма підстановка)
    z = np.zeros(n)
    for k in range(n):
        z[k] = (b[k] - sum(L[k, j] * z[j] for j in range(k))) / L[k, k]
        
    # UX = Z (Зворотна підстановка)
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - sum(U[k, j] * x[j] for j in range(k + 1, n))
        
    return x

def refine_solution(A, L, U, b, x0, eps_target=1e-14, max_iter=100):
    x = x0.copy()
    for i in range(max_iter):
        # Обчислення вектора нев'язки R = B - AX
        r = b - np.dot(A, x)
        
        # Перевірка умови зупинки за нормою нев'язки
        norm_r = np.max(np.abs(r))
        if norm_r < eps_target:
            return x, i
        
        # Розв'язуємо A * delta_x = R через вже готовий LU
        delta_x = solve_lu(L, U, r, len(b))
        x = x + delta_x
        
    return x, max_iter

#  Виконання 
n = generate_data(100)
A = np.loadtxt('matrix_A.txt')
b = np.loadtxt('vector_B.txt')

# LU-розклад
L, U = lu_decomposition(A, n)

# Початковий розв'язок
x_initial = solve_lu(L, U, b, n)
initial_error = np.max(np.abs(np.dot(A, x_initial) - b))

# Уточнення
x_refined, iterations = refine_solution(A, L, U, b, x_initial)
final_error = np.max(np.abs(np.dot(A, x_refined) - b))

print(f"Початкова похибка: {initial_error:.2e}")
print(f"Уточнена похибка: {final_error:.2e}")
print(f"Кількість ітерацій уточнення: {iterations}")