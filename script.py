import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model(x, t, b):
    s = x[0]
    i = x[1]
    dsdt = -b * s * i
    didt = -dsdt - (k * i)
    return dsdt, didt


k = 1 / 12.39
beta = 0.155
x0 = [1, 1e-6]
normal_contact = 2
mask_reduction = 0.75
lockdown_reduction = 0.6

t = np.linspace(0, 500)
days = np.arange(0, 500)
cap = np.full((1, 500), 0.1)[0]

b = normal_contact * beta
_, y1 = odeint(model, x0, t, args=(b,)).T
b = (normal_contact * beta * 0.5) + (normal_contact * beta *
                                     (1 - mask_reduction) * 0.5)
_, y2 = odeint(model, x0, t, args=(b,)).T
b = (normal_contact * beta * 0.25) + (normal_contact * beta *
                                      (1 - mask_reduction) * 0.75)
_, y3 = odeint(model, x0, t, args=(b,)).T
b = normal_contact * beta * (1 - lockdown_reduction)
_, y4 = odeint(model, x0, t, args=(b,)).T

plt.plot(t, y1, 'r', linewidth=2, label="No Distancing")
plt.plot(t, y2, 'b', linewidth=2, label="50% Mask Wearing")
plt.plot(t, y3, 'c', linewidth=2, label="75% Mask Wearing")
plt.plot(t, y4, 'k', linewidth=2, label="Lockdown")
plt.plot(days, cap, linestyle='dashed', linewidth=2, label="Hospital Capacity")
plt.title("COVID-19 SIR Models", loc='center', pad=15)
plt.xlabel("Time (Days)")
plt.ylabel("% Population with COVID-19")
plt.legend()
plt.show()
