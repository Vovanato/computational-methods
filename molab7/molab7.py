import numpy as np
import os

# --- 1. ГЕНЕРАЦІЯ ТА ЗАПИС ДАНИХ ---

def generate_and_save_data(n=100, exact_val=2.5):

    # Генеруємо випадкову матрицю
    A = np.random.rand(n, n)
    
    # Забезпечуємо діагональне переважання (достатня умова збіжності методу Якобі)
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + np.random.rand() + 1.0 
        
    # Точний розв'язок (всі x_i = 2.5)
    x_exact = np.full(n, exact_val)
    
    # Обчислення вектора вільних членів b = A * x_exact
    b = np.dot(A, x_exact)
    
    # Запис у текстові файли
    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_b.txt', b)
    
    print(f"Матриця {n}x{n} та вектор b успішно згенеровані та збережені.")

# --- 2. ДОПОМІЖНІ ФУНКЦІЇ ---

def read_matrix(filename):
    return np.loadtxt(filename)

def read_vector(filename):
    return np.loadtxt(filename)

def matrix_vector_product(A, x):
    return np.dot(A, x)

def vector_norm(v):
    # Використовуємо максимум модуля (нескінченна норма)
    return np.max(np.abs(v))

def matrix_norm(A):
    # Норма матриці: максимальна сума модулів елементів у рядку
    return np.max(np.sum(np.abs(A), axis=1))

# --- 3. ІТЕРАЦІЙНІ МЕТОДИ ---

def simple_iteration_method(A, b, x0, eps):
   # Метод простої ітерації
    n = len(b)
    x = x0.copy()
    
    # Вибір параметра tau згідно з умовою збіжності: 0 < tau < 2/||A||
    tau = 1.0 / matrix_norm(A) 
    
    iters = 0
    while True:
        # X^(k+1) = X^(k) - tau * (A*X^(k) - b)
        x_new = x - tau * (matrix_vector_product(A, x) - b)
        iters += 1
        
        # Умова зупинки: ||X^(k+1) - X^(k)|| < eps
        if vector_norm(x_new - x) < eps:
            break
            
        x = x_new.copy()
        
    return x_new, iters

def jacobi_method(A, b, x0, eps):
   # Метод Якобі
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)
    iters = 0
    
    while True:
        for i in range(n):
            # Сума a_ij * x_j^(k) для j != i
            s = sum(A[i, j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i, i]
            
        iters += 1
        if vector_norm(x_new - x) < eps:
            break
            
        x = x_new.copy()
        
    return x_new, iters

def seidel_method(A, b, x0, eps):
    #Метод Гауса-Зейделя
    n = len(b)
    x = x0.copy()
    iters = 0
    
    while True:
        x_old = x.copy()
        for i in range(n):
            # Сума з новими значеннями (j < i)
            s1 = sum(A[i, j] * x[j] for j in range(i))
            # Сума зі старими значеннями (j > i)
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            
            x[i] = (b[i] - s1 - s2) / A[i, i]
            
        iters += 1
        if vector_norm(x - x_old) < eps:
            break
            
    return x, iters

# --- 4. ГОЛОВНИЙ БЛОК ВИКОНАННЯ ---
if __name__ == "__main__":
    # 1. Генерація даних
    generate_and_save_data(n=100, exact_val=2.5)
    
    print("\n--- ДЕМОНСТРАЦІЯ РОБОТИ ДОПОМІЖНИХ ФУНКЦІЙ ---")
    
    # Виклик функцій зчитування
    A = read_matrix('matrix_A.txt')
    b = read_vector('vector_b.txt')
    print(f"Зчитування успішне! Розмірність матриці А: {A.shape}, вектора b: {b.shape}")
    
    # Виклик функцій обчислення норм
    norm_A = matrix_norm(A)
    norm_b = vector_norm(b)
    print(f"Норма матриці А: {norm_A:.4f}")
    print(f"Норма вектора b: {norm_b:.4f}")
    
    # Демонстрація добутку матриці на вектор (візьмемо для прикладу вектор з одиниць)
    test_vector = np.ones(100)
    product_result = matrix_vector_product(A, test_vector)
    print(f"Добуток матриці А на тестовий вектор (перші 3 елементи): {product_result[:3]}")
    
    print("\n" + "="*50)
    
    # 3. Початкові умови для методів
    n = len(b)
    x0 = np.full(n, 1.0) # Початкове наближення x_i^(0) = 1.0
    eps = 1e-14          # Задана точність eps_0 = 10^-14
    
    print(f"\nРозв'язок СЛАР (n={n}, eps={eps}):")
    print("-" * 50)
    
    # Виклик функції методу простої ітерації
    x_simple, iters_simple = simple_iteration_method(A, b, x0, eps)
    print(f"1. Метод простої ітерації:")
    print(f"   Кількість ітерацій: {iters_simple}")
    print(f"   Похибка (макс. відхилення від 2.5): {np.max(np.abs(x_simple - 3)):.2e}\n")
    
    # Виклик функції методу Якобі
    x_jacobi, iters_jacobi = jacobi_method(A, b, x0, eps)
    print(f"2. Метод Якобі:")
    print(f"   Кількість ітерацій: {iters_jacobi}")
    print(f"   Похибка (макс. відхилення від 2.5): {np.max(np.abs(x_jacobi - 3)):.2e}\n")
    
    # Виклик функції методу Зейделя
    x_seidel, iters_seidel = seidel_method(A, b, x0, eps)
    print(f"3. Метод Гауса-Зейделя:")
    print(f"   Кількість ітерацій: {iters_seidel}")
    print(f"   Похибка (макс. відхилення від 2.5): {np.max(np.abs(x_seidel - 3)):.2e}\n")