import math
import matplotlib.pyplot as plt

# 1. Задаємо функцію вологості ґрунту та її точну похідну
def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def exact_derivative(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

# Функція для чисельного диференціювання (центральна різниця)
def central_diff(t, h):
    if h == 0: return 0
    return (M(t + h) - M(t - h)) / (2 * h)

def main():
    t0 = 1.0
    exact_val = exact_derivative(t0)
    
    print("--- 1. Аналітичне розв'язання ---")
    print(f"Точне значення похідної y'(x0) = {exact_val:.10f}\n")

    # --- КРОК 2: Дослідження залежності похибки від кроку h ---
    print("--- 2. Дослідження похибки чисельного диференціювання ---")
    
    # Генеруємо кроки від 10^-20 до 10^3 (використовуємо дрібніші кроки для плавного графіка)
    h_values = [10**(i/2) for i in range(-40, 7)]
    
    h_plot = []
    error_plot = []
    
    min_error = float('inf')
    h0_opt = 0

    print(f"{'Крок h':<10} | {'Похідна y_0(h)':<20} | {'Похибка R':<20}")
    print("-" * 55)
    for h in h_values:
        if h == 0: continue
        approx_val = central_diff(t0, h)
        error = abs(approx_val - exact_val)
        
        # Додаємо дані для графіка (уникаємо нульової похибки для логарифмічної шкали)
        if error > 0:
            h_plot.append(h)
            error_plot.append(error)
            
        # Виводимо в консоль лише цілі степені для наочності
        if math.log10(h).is_integer() and 10**-16 <= h <= 10**1:
            print(f"{h:<10.1e} | {approx_val:<20.10f} | {error:<20.10e}")
            
        if error < min_error:
            min_error = error
            h0_opt = h

    print(f"\nОптимальний крок h0: {h0_opt:.1e}")
    print(f"Найкраща досягнута точність R0: {min_error:.10e}\n")

    # --- КРОК 3, 4, 5: Використання фіксованого кроку h = 10^-3 ---
    h_fixed = 1e-3
    print(f"--- 3-5. Фіксований крок h = {h_fixed:.1e} ---")
    y_prime_h = central_diff(t0, h_fixed)
    y_prime_2h = central_diff(t0, 2 * h_fixed)
    y_prime_4h = central_diff(t0, 4 * h_fixed)
    
    R1 = abs(y_prime_h - exact_val)
    print(f"Похибка при кроці h (R1): {R1:.10e}\n")

    # --- КРОК 6: Метод Рунге-Ромберга ---
    print("--- 6. Метод Рунге-Ромберга ---")
    y_prime_RR = y_prime_h + (y_prime_h - y_prime_2h) / 3
    R2 = abs(y_prime_RR - exact_val)
    print(f"Уточнене значення y_R' = {y_prime_RR:.10f}")
    print(f"Похибка методу Р-Р (R2): {R2:.10e}\n")

    # --- КРОК 7: Метод Ейткена ---
    print("--- 7. Метод Ейткена ---")
    numerator = y_prime_2h**2 - y_prime_4h * y_prime_h
    denominator = 2 * y_prime_2h - (y_prime_4h + y_prime_h)
    
    if denominator != 0:
        y_prime_E = numerator / denominator
        R3 = abs(y_prime_E - exact_val)
        ratio = (y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h)
        p = (1 / math.log(2)) * math.log(abs(ratio))
        
        print(f"Уточнене значення y_E' = {y_prime_E:.10f}")
        print(f"Похибка методу Ейткена (R3): {R3:.10e}")
        print(f"Оцінений порядок точності p = {p:.5f}\n")

    # --- ПОБУДОВА ГРАФІКА ---
    print("Будуємо графік... (Закрийте вікно графіка, щоб завершити програму)")
    
    plt.figure(figsize=(10, 6))
    plt.loglog(h_plot, error_plot, linewidth=2, color='#2c3e50', label="Загальна похибка R(h)")
    plt.loglog([h0_opt], [min_error], marker='o', markersize=8, color='#e74c3c', 
               label=f"Оптимальний крок: h ≈ {h0_opt:.1e}")
    
    plt.title("Залежність похибки чисельного диференціювання від кроку h", fontsize=14, pad=15)
    plt.xlabel("Крок сітки h (логарифмічна шкала)", fontsize=12)
    plt.ylabel("Абсолютна похибка R (логарифмічна шкала)", fontsize=12)
    
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.savefig("error_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()