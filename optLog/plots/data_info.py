import numpy as np
import pandas as pd
from scipy.signal import find_peaks as peaks
import matplotlib.pyplot as plt


class DataInfo:

    def __init__(self):
        self.n_points = 50

    def new_reference(self, signal, ref):
        new_ref = sum(signal[-self.n_points:]) / len(signal[-self.n_points:])
        if new_ref < 0:
            new_ref *= -1
        return new_ref

        variance = 0.0
        # half = int(len(signal) / 2)
        half = len(signal)-1
        # print("half = ", signal[half])
        s = slice(half-self.n_points, half)

        new_ref = sum(signal[s])/len(signal[s])

        for force in signal[s]:
            variance = variance + (force - new_ref)**2

        std_ref = (variance/len(signal[s]))**0.5

        # print("std_ref = ", std_ref)

        # if ref == 0.0:
        #     return ref

        if std_ref > 0.2:
            return ref
            # return -1
        else:
            return new_ref

    def moving_average(self, signal, N = 10):
        n = pd.Series(signal).rolling(window=N).mean().iloc[N-1:].values
        n = np.concatenate((np.array([s for s in signal[:N-1]]), n))

        return n

    # Overshoot
    def overshoot(self, signal, ref):
        # overshoot = 0.0
        signal = np.sign(signal[-1])*signal
        overshoot = max(signal)

        # for idx in range(len(signal)-1):
        #     if signal[idx] > new_ref and signal[idx] > overshoot:
        #         if signal[idx] != 0.0 and signal[idx] > 0.15*ref:
        #             if abs(signal[idx] - signal[idx-1]) < 0.02*signal[idx] and abs(signal[idx] - signal[idx+1]) < 0.02*signal[idx]:
        #                 overshoot = signal[idx]
        #                 break
        # if new_ref == -1:
        #     new_ref = ref

        if overshoot == 0.0:
            # print("foi o primeiro com new_ref = ", new_ref)
            # print("e o max de forca = ", max(signal))
            overshoot = abs(max(signal) - ref)/ref
        else:
            # print("foi o segundo com new_ref = ", new_ref)
            # print("e a forca de ref pro overshoot = ", overshoot)
            if ref != 0:
                overshoot = (overshoot - ref)/ref
            else:
                overshoot = np.maximum(abs(max(signal)), abs(min(signal)))

        return overshoot

    # Settling time
    def settling_time(self, signal, time, ref, error=0.05):
        ts = time[-1]
        signal = self.moving_average(signal)
        error_degree = 0.5*np.pi/180

        if abs(ref-signal[-1]) > error:
            ref = self.new_reference(signal, 1)

        # print("ref = ", ref)

        # if new_ref == -1:
        #     new_ref = 30

        # TODO: review this guy
        for element, t in zip(signal, time):
            if ref == 0:
                if abs(element) > error_degree:  # error * max(signal):
                    ts = t
                # print("")
            else:
                if abs(element - ref) > error * ref:
                    ts = t

        return ts

    # Steady state error
    def error_ss(self, signal, ref):
        last_value = sum(signal[-self.n_points:]) / len(signal[-self.n_points:])
        return abs(ref - last_value)

    # Total Variation
    def total_variation(self, signal):
        signal = self.moving_average(signal)
        p, _ = peaks(signal)
        v, _ = peaks(-signal)
        new_p = np.array([])
        new_v = np.array([])

        if len(p) == len(v):
            new_p = p[:]
            new_v = np.insert(v, 0, 0)[:-1]
        if len(p) > len(v):
            new_p = p[:-1]
            new_v = np.insert(v, 0, 0)
            # else: #len(p) < len(v)
            #     new_v = np.insert(v[:-1], 0, 0)[:-1]

        if new_p.shape[0] != 0 or new_v.shape[0] != 0:
            # check the for below and you will see this works
            tv = np.sum(np.abs(signal[new_p] - signal[v])) + np.sum(np.abs(signal[new_v] - signal[p]))
        else:
            tv = np.abs(signal[-1] - signal[0])

        # tv = abs(signal[p[0]] - signal[0])
        # for k in range(len(p)):
        #     if k > 0:
        #         tv += abs(signal[p[k]] - signal[v[k - 1]]) + abs(signal[v[k]] - signal[p[k]])
        #     else:
        #         tv += abs(signal[p[k]] - signal[0]) + abs(signal[p[k]] - signal[v[k]])

        return tv

    def rise_time(self, signal, time, ref):
        if signal[-1] < 0:
            new_signal = -1*signal
        else:
            new_signal = signal

        tr = 0
        error_degree = 0.5 * np.pi / 180
        if ref != 0:
            for k, value in enumerate(new_signal):
                if value - .9*ref > 0:
                    tr = time[k-1]
                    break
        else:
            for k, value in enumerate(new_signal):
                if value - error_degree > 0:
                    tr = time[k-1]
                    break
        return tr

    # TODO: min RMS tau (transparencia)
    # TODO: min TV qvel (robustez)

    def rms(self, signal):
        signal = self.moving_average(signal)
        return np.sqrt(np.mean(signal**2))

    def get_time(self, dt):
        time = np.cumsum(dt/1e3)
        return time[-1]

if __name__ == '__main__':
    data = pd.read_csv('../optLog/gustavo_const/g00_i00_K9.12_B45.99.csv')
    data_info = DataInfo()
    tv = data_info.total_variation(data["tau [Nm]"], list(range(len(data["tau [Nm]"]))))