import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math

# ПУНКТИ 1-3: Функція та базовий метод Сімпсона
def f(x):
    # Задана функція навантаження на сервер
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

# Точне значення інтегралу I_0 (Еталон)
I_0, _ = quad(f, a, b)
print("="*60)
print(f"Точне значення інтегралу I_0: {I_0:.12f}")
print("="*60)

# Складова квадратурна формула Сімпсона
def simpson_integral(f, a, b, N):
    if N % 2 != 0:
        N += 1  # Кількість відрізків має бути парною
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    
    integral = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return integral * h / 3

# ПУНКТ 4: Дослідження залежності точності від N
N_values = np.arange(10, 1002, 2)
errors = []
N_opt = None
eps_opt = None
target_eps = 1e-12

for N in N_values:
    I_N = simpson_integral(f, a, b, N)
    err = abs(I_N - I_0)
    errors.append(err)
    
    if N_opt is None and err <= target_eps:
        N_opt = N
        eps_opt = err

print(f"\nОптимальне N_opt (точність <= 1e-12): {N_opt}")
print(f"Похибка при N_opt (eps_opt): {eps_opt:.2e}")

# ПУНКТ 5: Обчислення похибки для N_0
# N_0 вибираємо кратним 8
N_0_base = N_opt // 10
N_0 = N_0_base + (8 - N_0_base % 8) if N_0_base % 8 != 0 else N_0_base
if N_0 == 0: N_0 = 8 # Захист, якщо N_opt дуже мале

I_N0 = simpson_integral(f, a, b, N_0)
eps_0 = abs(I_N0 - I_0)
print(f"\nБазове число розбиттів N_0: {N_0}")
print(f"Похибка при N_0 (eps_0): {eps_0:.2e}")

# ПУНКТИ 6-8: Уточнення результатів (Рунге-Ромберг та Ейткен)
print("\n" + "="*60)
print("УТОЧНЕННЯ РЕЗУЛЬТАТІВ")
print("="*60)

# Метод Рунге-Ромберга
I_N0_half = simpson_integral(f, a, b, N_0 // 2)
I_R = I_N0 + (I_N0 - I_N0_half) / 15
eps_R = abs(I_R - I_0)

print(f"\n--- Метод Рунге-Ромберга ---")
print(f"I(N0/2) [N={N_0//2}]: {I_N0_half:.12f}")
print(f"Уточнене значення I_R: {I_R:.12f}")
print(f"Похибка (eps_R): {eps_R:.2e}")

# Метод Ейткена
I_N0_quarter = simpson_integral(f, a, b, N_0 // 4)
I1, I2, I3 = I_N0, I_N0_half, I_N0_quarter  

denominator = 2 * I2 - (I1 + I3)
if denominator != 0:
    I_E = (I2**2 - I1 * I3) / denominator
    ratio = (I3 - I2) / (I2 - I1) if (I2 - I1) != 0 else 0
    p = (1 / math.log(2)) * math.log(abs(ratio)) if ratio != 0 else float('nan')
    eps_E = abs(I_E - I_0)

    print(f"\n--- Метод Ейткена ---")
    print(f"I(N0/4) [N={N_0//4}]: {I3:.12f}")
    print(f"Уточнене значення I_E: {I_E:.12f}")
    print(f"Оцінка порядку точності p: {p:.4f}")
    print(f"Похибка (eps_E): {eps_E:.2e}")

# Аналіз похибок
print("\n--- Порівняння похибок (Пункт 8) ---")
print(f"Початкова похибка (N={N_0}): {eps_0:.2e}")
print(f"Похибка Рунге-Ромберга:    {eps_R:.2e}")
print(f"Похибка Ейткена:           {eps_E:.2e}")

# ПУНКТ 9: Адаптивний алгоритм
print("\n" + "="*60)
print("АДАПТИВНИЙ АЛГОРИТМ (ЗМІННИЙ КРОК)")
print("="*60)

def adaptive_simpson_step(f, a, b, delta, fa, fc, fb):
    c = (a + b) / 2
    h = b - a
    d = (a + c) / 2
    e = (c + b) / 2
    
    fd = f(d)
    fe = f(e)
    evals = 2 
    
    I1 = (h / 6) * (fa + 4 * fc + fb)
    I2_left = (h / 12) * (fa + 4 * fd + fc)
    I2_right = (h / 12) * (fc + 4 * fe + fb)
    I2 = I2_left + I2_right
    
    if abs(I1 - I2) <= delta:
        return I2, evals
    else:
        left_I, left_evals = adaptive_simpson_step(f, a, c, delta / 2, fa, fd, fc)
        right_I, right_evals = adaptive_simpson_step(f, c, b, delta / 2, fc, fe, fb)
        return left_I + right_I, evals + left_evals + right_evals

def adaptive_simpson(f, a, b, delta):
    c = (a + b) / 2
    fa, fb, fc = f(a), f(b), f(c)
    integral, evals = adaptive_simpson_step(f, a, b, delta, fa, fc, fb)
    return integral, evals + 3

deltas = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

print(f"{'Параметр delta':<15} | {'Інтеграл':<18} | {'Похибка':<12} | {'Викликів f(x)'}")
print("-" * 65)

for delta in deltas:
    I_adapt, evals = adaptive_simpson(f, a, b, delta)
    err = abs(I_adapt - I_0)
    print(f"{delta:<15.0e} | {I_adapt:<18.12f} | {err:<12.2e} | {evals}")

# ПОБУДОВА ГРАФІКА
plt.figure(figsize=(10, 6))
plt.plot(N_values, errors, label=r'Похибка $\epsilon(N)$')
plt.axhline(y=target_eps, color='r', linestyle='--', label='Задана точність 1e-12')
plt.yscale('log')
plt.title('Залежність точності обчислення інтегралу від числа розбиттів N')
plt.xlabel('Число розбиттів (N)')
plt.ylabel('Похибка (логарифмічний масштаб)')
plt.grid(True)
plt.legend()
plt.show()