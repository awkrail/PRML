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
# plt.scatter(x, t)
# plt.plot(true_x, w0_curve, color="#4daf4a") 
# plt.show()

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
# plt.scatter(x, t)
# plt.plot(x, a_fit*x+b_fit, color="#4daf4a")
# plt.show()

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
# plt.scatter(x, t)
#plt.plot(true_x,result[0][0] + result[0][1]*true_x + result[0][2]*(true_x**2) + result[0][3]*(true_x**3) , color="#4daf4a")
# plt.plot(true_x, true_model, color="r")
# plt.show()


# M == 9
def fit_func9(params, x, y):
    new_x = [[data**i for i in range(9)] for data in x]
    new_w = np.array(params, dtype=np.float32)
    new_x = np.array(new_x, dtype=np.float32)
    residual = y - new_w @ new_x.T
    return residual
"""
result = [0. for _ in range(9)]
# result = optimize.leastsq(fit_func9, x, t, p0=result)
plt.clf()
plt.scatter(x, t)
np_result = np.array(result[0], dtype=np.float32)
truex_arr = [[dt**i for i in range(9)] for dt in true_x]
np_truex = np.array(truex_arr, dtype=np.float32)
result_y = [true_x @ np_result for true_x in np_truex]
plt.plot(true_x, result_y, color="#4daf4a")
plt.plot(true_x, true_model, color="r")
plt.show()
"""

# M == 9 : guchoku
def fit_func9_g(params, x, y):
   w0 = params[0]
   w1 = params[1]
   w2 = params[2]
   w3 = params[3]
   w4 = params[4]
   w5 = params[5]
   w6 = params[6]
   w7 = params[7]
   w8 = params[8]
   residual = y - (w0 + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4) + \
        w5*(x**5) + w6*(x**6) + w7*(x**7) + w8*(x**8))
   return residual

result = [0. for _ in range(9)]
result = optimize.leastsq(fit_func9_g, result, args=(x, t))
plt.clf()
plt.scatter(x, t)
data = result[0]
plt.plot(true_x, data[0] + data[1]*true_x + data[2]*(true_x**2) + data[3]*(true_x**3) + \
        data[4]*(true_x**4) + data[5]*(true_x**5) + data[6]*(true_x**6) + data[7]*(true_x**7) + \
        data[8]*(true_x**8), color="#4daf4a")
plt.plot(true_x, true_model, color="r")
plt.show()










