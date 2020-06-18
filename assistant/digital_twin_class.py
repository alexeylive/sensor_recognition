import numpy as np
from scipy.interpolate import griddata


class DigitalTwin:
    def __init__(self, sensor_info):
        self.gen = None
        self.load_input()

        self.X = self.gen[:, 0]
        self.Y = self.gen[:, 1]
        self.EX = self.gen[:, 2]
        self.EY = self.gen[:, 3]
        self.GXY = self.gen[:, 4]

        self.ideal_sensor = self.force_sensor_output_signal([10, 30, 10, 30], [0, 0, 0, 0], [0, 0, np.pi, np.pi])

        self.current_sensor = sensor_info

    def force_sensor_output_signal(self, X, Y, phi):
        N = 4
        r_deform = []
        r_undeform = 350
        gauge_factor = 2

        for i in range(N):
            r_deform.append(r_undeform * np.exp(gauge_factor * self.strain_gauge_strain(X[i], Y[i], phi[i])))

        output_signal = (r_deform[1] / (r_deform[0] + r_deform[1]) - r_deform[2] / (r_deform[2] + r_deform[3]))
        return np.abs(output_signal)

    def strain_gauge_strain(self, x_c, y_c, phi):
        x_c = np.abs(x_c)
        y_c = np.abs(y_c)
        ex_prime = self.EX * np.cos(phi) ** 2 + self.EY * np.sin(phi) ** 2 + 0.5 * self.GXY * np.sin(2 * phi)
        return griddata((self.X, self.Y), ex_prime, (x_c, y_c))

    def get_sensor_result(self):
        current_sensor = self.force_sensor_output_signal(self.current_sensor[0],
                                                         self.current_sensor[1],
                                                         self.current_sensor[2])

        return (self.ideal_sensor - current_sensor) / self.ideal_sensor * 100

    def load_input(self):
        input_matrix = np.genfromtxt('total.txt', delimiter=',')
        self.gen = input_matrix
