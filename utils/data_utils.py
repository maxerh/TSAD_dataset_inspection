import numpy as np


def rolling_window(a, window_size, step):
    nb_samples = (a.shape[0] - window_size)//step + 1
    return np.lib.stride_tricks.as_strided(
        a,
        shape=(nb_samples, window_size, a.shape[-1]),
        strides=(a.strides[0] * step, a.strides[0], a.strides[1])
    ).copy()


def load_data(dl, datasets, entities, exclude_anomaly=False):
    dict1 = {}
    for d in datasets:
        dict1[d] = {}
        for i, entity in enumerate(entities[d]):
            dl.load_dataset(d, entity)
            data = dl.data
            label = None
            relative_positions = None
            if dl.label is not None:
                label = dl.label
                total_labels = len(label)
                relative_positions = [idx / total_labels for idx in range(total_labels)]
                if exclude_anomaly:
                    data = data[label==0]   # just nominal data when having test data.

            dict1[d][entity] = {'data': data, 'label': label,
                                'min': np.min(data, axis=0),
                                'max': np.max(data, axis=0),
                                'mean': np.mean(data, axis=0),
                                'std': np.std(data, axis=0),
                                "relative_positions": relative_positions,
                                }
    return dict1

def find_segments(data: np.ndarray):
    changes = np.diff(data, prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    segments = np.column_stack((starts, ends)).tolist()
    return segments