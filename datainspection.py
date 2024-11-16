import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


class DataInspector:
    def __init__(self, lags, vis):
        self.lags = lags
        self.vis_counter = 0
        self.vis = vis

    def plot_acf_one_channel(self, data, channel):
        plot_acf(data[:, channel], lags=self.lags)
        plt.show()

    def get_acf(self, data):
        if isinstance(data, list):
            data = data[0]
        acf_array = np.ones([data.shape[1], self.lags + 1])
        minmax_count = 0
        i = 0
        for c in range(data.shape[1]):
            if not min(data[:, c]) == max(data[:, c]):
                acf_array[i] = acf(data[:, c], nlags=self.lags)
                i += 1
            else:
                minmax_count += 1
        if minmax_count > 0:
            acf_array = acf_array[:-minmax_count, :]
        return acf_array

    def plot_acf_all_channels(self, data):
        acf_array = self.get_acf(data)
        plt.imshow(acf_array, aspect='auto')
        self.vis.plot_correlation(self.vis_counter, acf_array,
                                  f"{self.vis_counter}: min==max: {data[0].shape[1] - acf_array.shape[0]}")

        #plt.title(f"min==max: {data[0].shape[1]-acf_array.shape[0]}")
        #plt.show()

    def plot_mean_acf(self, data, entity):
        acf_array = self.get_acf(data)
        merged_acf_array = np.mean(acf_array, axis=0)
        self.vis.plot_mean_acf(self.vis_counter, merged_acf_array, entity)
        return merged_acf_array

    def plot_acf_all_machines_all_channels(self, data):
        acf_array = self.get_acf(data)
        self.vis.plot_correlation(self.vis_counter, acf_array, f"{self.vis_counter}: min==max: {data[0].shape[1]-acf_array.shape[0]}")
        self.vis_counter += 1

    def show_correlations(self):
        self.vis.show_plot()

    def draw_correlations(self):
        self.vis.draw_plot()