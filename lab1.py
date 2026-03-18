import requests 
import numpy as np
import matplotlib.pyplot as plt 

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106" 

print("Виконуємо запит до API...")
response = requests.get(url) 
data = response.json() 
results = data["results"] 
n = len(results) 

print("Кількість вузлів:", n) 

def haversine(lat1, lon1, lat2, lon2): 
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2) 
    dphi = np.radians(lat2 - lat1) 
    dlambda = np.radians(lon2 - lon1) 
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2 
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 

coords = [(p["latitude"], p["longitude"]) for p in results] 
elevations = [p["elevation"] for p in results] 

distances = [0] 
for i in range(1, n): 
    d = haversine(*coords[i-1], *coords[i]) 
    distances.append(distances[-1] + d) 

print("\nТабуляція (відстань, висота):") 
print(" | Distance (m) | Elevation (m) |") 
for i in range(n): 
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f} |") 

# Будуємо графік сирих (дискретних) даних
plt.figure(figsize=(10, 5))
plt.plot(distances, elevations, 'o', label='Дискретні вузли (API)', color='red')
plt.title('Профіль висоти маршруту: Заросляк - Говерла')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()
plt.show()

#  Метод прогонки для знаходження коефіцієнтів C
x = np.array(distances)
y = np.array(elevations)
N = len(x)

# Знаходимо кроки h (різниця між сусідніми x)
h = np.zeros(N - 1)
for i in range(N - 1):
    h[i] = x[i+1] - x[i]

# Ініціалізуємо масиви для коефіцієнтів системи рівнянь (alpha, beta, gamma, delta)
alpha = np.zeros(N)
beta = np.zeros(N)
gamma = np.zeros(N)
delta = np.zeros(N)

# Заповнюємо матрицю для внутрішніх вузлів
for i in range(1, N - 1):
    alpha[i] = h[i-1]
    beta[i] = 2 * (h[i-1] + h[i])
    gamma[i] = h[i]
    delta[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

#ПРЯМА ПРОГОНКА
A = np.zeros(N)
B = np.zeros(N)

for i in range(1, N - 1):
    denominator = alpha[i] * A[i-1] + beta[i]
    A[i] = -gamma[i] / denominator
    B[i] = (delta[i] - alpha[i] * B[i-1]) / denominator

# ЗВОРОТНА ПРОГОНКА
c = np.zeros(N)

c[0] = 0 
c[N-1] = 0


for i in range(N - 2, 0, -1):
    c[i] = A[i] * c[i+1] + B[i]

print("\nЗнайдені коефіцієнти c_i:")
for i in range(N):
    print(f"c[{i:2d}] = {c[i]:.6f}")
    # Крок 9: Обчислення коефіцієнтів a, b, d
a = np.zeros(N - 1)
b = np.zeros(N - 1)
d = np.zeros(N - 1)

for i in range(N - 1):
    # a_i дорівнює значенню функції на початку інтервалу
    a[i] = y[i] 
    
    # d_i виражається через різницю сусідніх c_i
    d[i] = (c[i+1] - c[i]) / (3 * h[i]) 
    
    # b_i обчислюється з умови неперервності
    b[i] = (y[i+1] - y[i]) / h[i] - (h[i] / 3) * (c[i+1] + 2 * c[i])

print("\nКоефіцієнти сплайнів (a, b, c, d) для кожного інтервалу:") #
for i in range(N - 1):
    print(f"Інтервал {i:2d}: a={a[i]:8.2f}, b={b[i]:8.4f}, c={c[i]:8.4f}, d={d[i]:8.6f}")

# Крок 10 та 12: Побудова гладкого графіка
x_smooth = []
y_smooth = []

# Проходимо по кожному інтервалу і генеруємо на ньому багато точок для гладкості
for i in range(N - 1):
    # Створюємо 50 проміжних точок між x[i] та x[i+1]
    x_interval = np.linspace(x[i], x[i+1], 50)
    
    # Обчислюємо значення сплайна S_i(x) для кожної проміжної точки
    dx = x_interval - x[i]
    y_interval = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    
    x_smooth.extend(x_interval)
    y_smooth.extend(y_interval)

# Малюємо фінальний графік
plt.figure(figsize=(12, 6))
# Спочатку малюємо наші "сирі" точки з API
plt.plot(x, y, 'ro', label='Дискретні вузли (API)', zorder=5)
# Потім накладаємо наш згладжений маршрут
plt.plot(x_smooth, y_smooth, 'b-', label='Кубічний сплайн', linewidth=2, zorder=4)

plt.title('Профіль висоти маршруту: Заросляк - Говерла (Інтерполяція)')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()
plt.show()
#  ДОДАТКОВЕ ЗАВДАННЯ 
print("\n" + "="*40)
print(" ДОДАТКОВІ ХАРАКТЕРИСТИКИ МАРШРУТУ")
print("="*40)

# 1. Загальна довжина та перепади висот
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}") 

# Сумарний набір висоти: додаємо лише ті ділянки, де висота зростає
total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, N))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}") 

# Сумарний спуск: додаємо лише ті ділянки, де висота падає
total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, N)) 
print(f"Сумарний спуск (м): {total_descent:.2f}") 

# 2. Аналіз градієнта (через похідну сплайна)
xx = np.array(x_smooth)
yy_full = np.array(y_smooth)

xx_unique, unique_indices = np.unique(xx, return_index=True)
yy_unique = yy_full[unique_indices]

# Обчислюємо градієнт на унікальних точках
grad_full = np.gradient(yy_unique, xx_unique) * 100

print("\n--- Аналіз крутизни (градієнта) ---")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

# 3. Механічна енергія підйому
# Звичайна шкільна фізика: Потенціальна енергія E = m * g * h
mass = 80  # маса людини у кг 
g = 9.81   # прискорення вільного падіння 
energy = mass * g * total_ascent 

print("\n--- Енерговитрати на підйом (для маси 80 кг) ---")
print(f"Механічна робота (Дж): {energy*5:.2f}") 
print(f"Механічна робота (кДж): {energy*5 / 1000:.2f}")
# 1 кілокалорія = 4184 Джоулі
print(f"Енергія (ккал): {energy*5 / 4184:.2f}") 