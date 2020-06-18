import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata




INPUT = np.genfromtxt('total.txt', delimiter=',')




X = INPUT[:, 0]

Y = INPUT[:, 1]

EX = INPUT[:, 2]

EY = INPUT[:, 3]

GXY = INPUT[:, 4]


def strain_gauge_strain(x_c, y_c, phi):
    x_c = np.abs(x_c)
    y_c = np.abs(y_c)

    ex_prime = EX*np.cos(phi)**2 + EY*np.sin(phi)**2 + 0.5*GXY*np.sin(2*phi)
    return griddata((X, Y), ex_prime, (x_c, y_c))


phi = np.linspace(0, np.pi * 2, 20)
eps = np.linspace(0, np.pi * 2, 20)

for i in range(len(phi)):
    eps[i] = strain_gauge_strain(25, 0, phi[i])


def force_sensor_output_signal(X, Y, phi):
    N = 4

    R_deform = []
    R_undeformed = 350
    gauge_factor = 2

    for i in range(4):
        R_deform.append(R_deform*np.exp(2 * gauge_factor * strain_gauge_strain(X[i], Y[i], phi[i])))

    output_signal = (R_deform[1] / (sum(R_deform[0]), R_deform[1])) - R_deform[2]/(sum(R_deform[3]), R_deform[4])
    return np.abs(output_signal)

print(force_sensor_output_signal([10, 30, 10, 30], [0,0,0,0], [0,0,np.pi, np.pi]))
print(force_sensor_output_signal([9.9, 30.1, 10.3, 29], [0,0,0,0], [np.pi/100, -np.pi/100, np.pi, np.pi]))


plt.polar(phi, eps, 'ro')
plt.show()

