import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

x = np.linspace(0, 1, 10)
t = np.sin(2 * np.pi * x) + np.random.randn() / 10.0
true_x = np.linspace(0, 1, 100)
true_model = np.sin(2 * np.pi * true_x)
print(t)
plt.clf()
plt.scatter(x, t)
plt.plot(true_x, true_model, color="#4daf4a")
plt.show()

# Mの数で変更

# M == 0
w0 = (1/10) * t.sum()
w0_curve = np.ones(100, dtype=np.float32) * w0
plt.clf()
plt.scatter(x, t)
plt.plot(true_x, w0_curve, color="#4daf4a") 
plt.show()

# M == 1
def fit_func(params, x, y):
    a = params[0]
    b = params[1]
    residual = y - (a * x + b)
    return residual

params1 = [0., 0.]
result = optimize.leastsq(fit_func, params1, args=(x, t))
print(result)
a_fit = result[0][0]
b_fit = result[0][1]
plt.clf()
plt.scatter(x, t)
plt.plot(x, a_fit*x+b_fit, color="#4daf4a")
plt.show()

# M == 3
def fit_func3(params, x, y):
    w0 = params[0]
    w1 = params[1]
    w2 = params[2]
    w3 = params[3]
    residual = y - (w0 + w1*x + w2*(x**2) + w3*(x**3))
    return residual


result = [0., 0., 0., 0.]
result = optimize.leastsq(fit_func3, result, args=(x, t))
plt.clf()
plt.scatter(x, t)
plt.plot(true_x,result[0][0] + result[0][1]*true_x + result[0][2]*(true_x**2) + result[0][3]*(true_x**3) , color="#4daf4a")
plt.show()












