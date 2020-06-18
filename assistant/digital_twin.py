import numpy as np
from scipy.interpolate import griddata

INPUT = np.genfromtxt('total.txt', delimiter=',')

print(INPUT.shape)

X = INPUT[:,0]

Y = INPUT[:,1]

EX = INPUT[:,2]

EY = INPUT[:,3]

GXY = INPUT[:,4]


def force_sensor_output_signal(X, Y, phi):
	N = 4
	r_deform = []
	r_undeform = 350
	gauge_factor = 2

	for i in range(4):
		r_deform.append(r_undeform*np.exp(gauge_factor*strain_gauge_strain(X[i], Y[i], phi[i])))

	output_signal = (r_deform[1]/(r_deform[0]+r_deform[1])-r_deform[2]/(r_deform[2]+r_deform[3]))
	return np.abs(output_signal)


def strain_gauge_strain(x_c, y_c, phi):
	x_c = np.abs(x_c)
	y_c = np.abs(y_c)

	ex_prime = EX*np.cos(phi)**2 + EY*np.sin(phi)**2 + 0.5*GXY*np.sin(2*phi)

	return griddata((X,Y), ex_prime, (x_c,y_c))


a = force_sensor_output_signal([10, 30, 10, 30], [0, 0, 0, 0], [0, 0, np.pi, np.pi])
b = force_sensor_output_signal([9.9, 30.1, 10.3, 29], [0, 0, 0, 0], [np.pi/100, -np.pi/100, np.pi-np.pi/100, np.pi])

dif = (a-b)/a*100
print(dif)

phi = np.linspace(0,np.pi*2,20)
eps = np.linspace(0,np.pi*2,20)

for i in range(len(phi)):
	eps[i] = strain_gauge_strain(25,0,phi[i])

import matplotlib.pyplot as plt

plt.polar(phi, eps, 'ro')
plt.show()
