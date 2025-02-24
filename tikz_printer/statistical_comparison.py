#from scipy.stats import energy_distance, kstest
import os
import pprint
import time
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from utils.data_utils import *
import matplotlib.pyplot as plt

def statistics(dl_train, dl_test, datasets, entities):
    distances = calc_energy_distance(dl_train, dl_test, datasets, entities)
    plot_distances(distances)

def plot_distances(distances):
    for ds in distances:
        if len(distances[ds].keys()) == 1:
            rows = 1
            cols = 1
        else:
            rows = 4
            cols = 3
        fig, axs = plt.subplots(rows, cols)
        max_plots = rows*cols
        idx_row = 0
        idx_col = 0
        for idx_entity, entity in enumerate(distances[ds]):
            if idx_entity >= max_plots:
                break
            entity_data = distances[ds][entity]
            bar_data = [
                entity_data["train1,train2"]["dist"],
                entity_data["train,test_n"]["dist"],
                entity_data["train,before"]["dist"],
                entity_data["train,after"]["dist"],
            ]
            p_values = [
                entity_data["train1,train2"]["p_value"],
                entity_data["train,test_n"]["p_value"],
                entity_data["train,before"]["p_value"],
                entity_data["train,after"]["p_value"],
            ]
            y_pos = np.arange(len(bar_data))
            if len(distances[ds].keys()) == 1:
                a = axs
            else:
                a = axs[idx_row, idx_col]
            a.bar(y_pos, bar_data, align='center')
            a.set_xticks(y_pos, ["train","test_n","bef","aft"])
            a.set_title(f'{entity}: {p_values}')

            idx_col += 1
            if idx_col >= cols:
                idx_col = 0
                idx_row += 1
        fig.suptitle(f'{ds}: Energy Distances between train data and ...')
        plt.subplots_adjust(hspace=0.4, wspace=0.25)
        plt.show()


def calc_distance(x, y=None, metric="euclidean"):
    if y is None:
        y = x
    dists = cdist(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1), metric=metric)
    return dists


def energy_distance(x, y, metric="euclidean"):
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0
    # Calculate pairwise distances
    d_xx = calc_distance(x, metric=metric)
    d_yy = calc_distance(y, metric=metric)
    d_xy = calc_distance(x, y, metric=metric)
    return 2 * np.mean(d_xy) - np.mean(d_xx) - np.mean(d_yy)


def perturbation_energy_test(P, Q, energy_distance=energy_distance, num_permutations=1000):

    # Get sizes and check for empty sets
    n_P, n_Q = len(P), len(Q)
    if n_P == 0 or n_Q == 0:
        return -1, -1

    # Compute observed Energy Distance
    observed_distance = energy_distance(P, Q)

    total_set = np.vstack([P,Q])
    perturbed_distances = np.zeros(num_permutations)
    for i in tqdm(range(num_permutations)):
        #indices = np.random.permutation(np.arange(n_P+n_Q))
        #pertubed_distances[i] = energy_distance(total_set[indices[:n_P]], total_set[indices[n_P:]])
        np.random.shuffle(total_set)
        perturbed_distances[i] = energy_distance(total_set[:n_P], total_set[n_P:])
    p_value = np.mean(perturbed_distances >= observed_distance)
    return float(observed_distance), float(p_value)


def monte_carlo_energy_test(P, Q, energy_distance=energy_distance, num_permutations=10000):
    """
    Monte Carlo Approximation for Energy Distance Significance Testing

    Parameters:
        P (array-like): sample set 1
        Q (array-like): sample set 2
        energy_distance (function): Function computing Energy Distance
        num_permutations (int): Number of Monte Carlo iterations

    Returns:
        observed_distance (float): Energy Distance between original P and Q
        p_value (float): Approximate p-value from Monte Carlo approximation
    """
    # Compute observed Energy Distance
    observed_distance = energy_distance(P, Q)

    # Get minimum sample sizes
    min_length = min(len(P), len(Q))
    if min_length == 0:
        return observed_distance, -1

    permuted_distances = np.zeros(num_permutations)
    for i in tqdm(range(num_permutations)):
        permuted_distances[i] = energy_distance(P[np.random.permutation(P.shape[0])[:min_length]],
                                                Q[np.random.permutation(Q.shape[0])[:min_length]])

    # Compute p-value: proportion of times permuted distances >= observed
    p_value = np.mean(np.array(permuted_distances) >= observed_distance)

    return observed_distance, p_value

def get_segments_before_and_after_anomaly(test, test_labels, window_size):
    anomaly_segements = find_segments(test_labels)
    test_labels_expended = np.zeros_like(test_labels)
    windows_before_anomaly = []
    windows_after_anomaly = []
    for seg in anomaly_segements:
        end1 = seg[0]
        start1 = end1 - window_size
        if sum(test_labels[start1:end1]) == 0:
            sample = test[start1:end1]
            if sample.shape[0] == window_size:
                windows_before_anomaly.append(sample)
        start2 = seg[1] + 1
        end2 = start2 + window_size
        if sum(test_labels[start2:end2]) == 0:
            sample = test[start2:end2]
            if sample.shape[0] == window_size:
                windows_after_anomaly.append(test[start2:end2])
        test_labels_expended[start1:end2] = 1
    if len(windows_before_anomaly) > 0:
        windows_before_anomaly = np.stack(windows_before_anomaly)
    else:
        windows_before_anomaly = np.zeros([0, window_size, test.shape[-1]])
    if len(windows_after_anomaly) > 0:
        windows_after_anomaly = np.stack(windows_after_anomaly)
    else:
        windows_after_anomaly = np.zeros([0, window_size, test.shape[-1]])
    return windows_before_anomaly, windows_after_anomaly, test_labels_expended


def calc_energy_distance(dl_train, dl_test, datasets, entities):
    """Rizzo, Szekely “Energy distance".”"""
    same_length = False
    data_dict_train = load_data(dl_train, datasets, entities)
    data_dict_test = load_data(dl_test, datasets, entities)
    distance_dict = dict()
    for d in datasets:
        distance_dict[d] = dict()
        filename = f'output/energy_distance/statistical_test_{d}.pkl'
        if os.path.isfile(filename):
            with open(filename, 'rb') as handle:
                distance_dict[d] = pickle.load(handle)
            break
        for i, entity in enumerate(entities[d]):
            window_size = 25
            step_size=window_size
            min_anomaly_time_steps = 10
            train = data_dict_train[d][entity]["data"]
            test_o = data_dict_test[d][entity]["data"]
            test_labels = data_dict_test[d][entity]["label"]
            test = np.copy(test_o)

            scaler = StandardScaler()
            scaler.fit(train)
            train = scaler.fit_transform(train)
            #test = scaler.transform(test)
            test = scaler.fit_transform(test)
            train_windows = rolling_window(train, window_size, window_size) # no overlap in training
            indices = np.arange(train_windows.shape[0])
            indices = np.random.permutation(indices)
            idx1 = indices[:len(indices)//2]
            idx2 = indices[len(indices)//2:]
            subset_train_1 = train_windows[idx1]
            subset_train_2 = train_windows[idx2]

            test_windows_masked_anomalies = test_o
            test_windows_masked_anomalies[test_labels==1] = 0
            #test_windows_masked_anomalies = scaler.transform(test_windows_masked_anomalies)
            test_windows_masked = rolling_window(test_windows_masked_anomalies, window_size, step_size)

            windows_before_anomaly, windows_after_anomaly, test_labels_expended = get_segments_before_and_after_anomaly(test, test_labels, window_size)

            test_windows = rolling_window(test, window_size, step_size)
            test_labels_windows = rolling_window(np.expand_dims(test_labels_expended, -1), window_size, step_size)

            test_windows_normal = test_windows[np.argwhere(np.sum(test_labels_windows[...,0], axis=1)==0)[:,0]]
            test_windows_abnormal = test_windows[np.argwhere(np.sum(test_labels_windows[...,0], axis=1)>=min_anomaly_time_steps)[:,0]]

            print("n samples:")
            print("train:", len(train_windows))
            print("train_subset:", len(subset_train_2))
            print("test_n:", len(test_windows_normal))
            print("before:", len(windows_before_anomaly))
            print("after:", len(windows_after_anomaly))

            if same_length:
                min_length = np.min([len(train_windows), len(subset_train_2), len(test_windows_normal), len(windows_before_anomaly), len(windows_after_anomaly)])
                train_windows = train_windows[:min_length]
                test_windows_normal = test_windows_normal[:min_length]
                test_windows_abnormal = test_windows_abnormal[:min_length]
                test_windows_masked = test_windows_masked[:min_length]
                windows_before_anomaly = windows_before_anomaly[:min_length]
                windows_after_anomaly = windows_after_anomaly[:min_length]

            #dist_train_test_normal = energy_distance(train_windows, test_windows_normal)
            #dist_train_test_abnormal = energy_distance(train_windows, test_windows_abnormal)
            #dist_test_normal_test_abnormal = energy_distance(test_windows_normal, test_windows_abnormal)
            #dist_train_test_masked = energy_distance(train_windows, test_windows_masked)
            #dist_train_subsets = energy_distance(subset_train_1, subset_train_2)
            #dist_train_before_a = energy_distance(train_windows, windows_before_anomaly)
            #dist_train_after_a = energy_distance(train_windows, windows_after_anomaly)
            #dist_before_after = energy_distance(windows_before_anomaly, windows_after_anomaly)

            #dist_train_test_abnormal, p_train_test_abnormal = monte_carlo_energy_test(train_windows, test_windows_abnormal)
            #dist_test_normal_test_abnormal, p_test_normal_test_abnormal = monte_carlo_energy_test(test_windows_normal, test_windows_abnormal)
            #dist_train_test_masked, p_train_test_masked = monte_carlo_energy_test(train_windows, test_windows_masked)
            #dist_train_test_normal, p_train_test_normal = monte_carlo_energy_test(train_windows, test_windows_normal, num_permutations=10)
            dist_train_subsets, p_train_subsets = perturbation_energy_test(subset_train_1, subset_train_2, num_permutations=10000)
            dist_train_test_normal, p_train_test_normal = perturbation_energy_test(train_windows, test_windows_normal, num_permutations=10000)
            dist_train_before_a, p_train_before_a = perturbation_energy_test(train_windows, windows_before_anomaly, num_permutations=10000)
            dist_train_after_a, p_train_after_a = perturbation_energy_test(train_windows, windows_after_anomaly, num_permutations=10000)
            #dist_before_after, p_before_after = monte_carlo_energy_test(windows_before_anomaly, windows_after_anomaly)

            distance_dict[d][entity] = {
                "train,test_n": {"dist": dist_train_test_normal, "p_value": p_train_test_normal},
                #"train,test_a": {"dist": dist_train_test_abnormal, "p_value": p_train_test_abnormal},
                #"test_n,test_a": {"dist": dist_test_normal_test_abnormal, "p_value": p_test_normal_test_abnormal},
                #"train,test_masked": {"dist": dist_train_test_masked, "p_value": p_train_test_masked},
                "train1,train2": {"dist": dist_train_subsets, "p_value": p_train_subsets},
                "train,after": {"dist": dist_train_after_a, "p_value": p_train_after_a},
                "train,before": {"dist": dist_train_before_a, "p_value": p_train_before_a},
                #"before,after": {"dist": dist_before_after, "p_value": p_before_after},
            }
        with open(filename, 'wb') as handle:
            pickle.dump(distance_dict[d], handle, protocol=pickle.HIGHEST_PROTOCOL)

    pprint.pprint(distance_dict)

    return distance_dict


