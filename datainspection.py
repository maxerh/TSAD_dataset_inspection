import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter('ignore', InterpolationWarning)


class DataInspectorStationary:
    def __init__(self):
        self.p = 0.05
        self.window_size = 52 # for weekly data


    def print_results(self, data, test_name):
        data = pd.DataFrame(data)
        if test_name == 'adf':
            output = adfuller(data)
        elif test_name == 'kpss':
            output = kpss(data)
        test_score = output[0]
        pval = output[1]
        lags = output[2]
        decision = 'Non-Stationary'
        if test_name == "adf":
            critical = output[4]
            if pval < self.p:
                decision = 'Stationary'
        elif test_name == "kpss":
            critical = output[3]
            if pval >= self.p:
                decision = 'Stationary'
        output_dict = {
            'Test Statistics': test_score,
            'p-value': pval,
            'Number of lags': lags,
            "decision": decision,
        }

        for k, v in critical.items():
            output_dict[f"Critical Value ({k})"] = v
        return pd.Series(output_dict, name=test_name)

    def check_stationary(self, data):
        kps = kpss(data)  # Kwiatkowski-Phillips-Schmidt-Shin Test
        adf = adfuller(data)  # Augmented Dickey-Fuller

        kpss_pv, adf_pv = kps[1], adf[1]
        kpssh, adfh = 'Stationary', 'Non-stationary'

        if adf_pv < self.p:
            # Reject ADF Null Hypothesis
            adfh = 'Stationary'
        if kpss_pv < self.p:
            # Reject KPSS Null-Hypothesis
            kpssh = 'Non-Stationary'
        return (kpssh, adfh)

    def get_methods(self, data):
        first_order_diff = data.diff().dropna()
        second_order_diff = data.diff(self.window_size).diff().dropna()
        subtract_rolling_mean = data - data.rolling(self.window_size)
        log_transform = np.log(data)
        decomp = seasonal_decompose(data)
        seasonal_detrend = decomp.observed - decomp.trend
        cyclic, trend = hpfilter(data)  # Hodrick-Prescott filter
        return [first_order_diff, second_order_diff, subtract_rolling_mean, log_transform, seasonal_detrend, cyclic]

    def plot_comparison(self, data, plot_type='line'):
        methods = self.get_methods(data)
        n = len(methods) // 2
        fig, ax = plt.subplots(n, 2, sharex=True, figsize=(20,10))
        for i, method in enumerate(methods):
            method.dropna(inplace=True)
            name = [n for n in globals() if globals()[n] in method]
            v, r = i // 2, i % 2
            kpss_s, adf_s = self.check_stationary(method)

            method.plpot(kind=plot_type,
                         ax=ax[v,r],
                         legend=False,
                         title=f"{name[0]} --> KPSS: {kpss_s}, ADF: {adf_s}")
            ax[v,r].title.set_size(20)
            method.rolling(self.window_size).mean().plot(ax=ax[v,r], legend=False)

class DataInspectorACF:
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