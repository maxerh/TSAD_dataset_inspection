import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kstest, pearsonr, spearmanr

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from utils.data_utils import *


class PFA(object):
    """
    Principal Feature Analysis
    """
    def __init__(self, diff_n_features=2, q=None, explained_var=0.95):
        self.q = q
        self.diff_n_features = diff_n_features
        self.explained_var = explained_var

    def fit(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA().fit(X)

        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i + 1]) for i in range(len(explained_variance))]
            for i, j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break

        A_q = pca.components_.T[:, :q]

        clusternumber = min([q + self.diff_n_features, X.shape[1]])

        kmeans = KMeans(n_clusters=clusternumber).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

    def fit_transform(self, X):
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA().fit(X)

        if not self.q:
            explained_variance = pca.explained_variance_ratio_
            cumulative_expl_var = [sum(explained_variance[:i + 1]) for i in range(len(explained_variance))]
            for i, j in enumerate(cumulative_expl_var):
                if j >= self.explained_var:
                    q = i
                    break

        A_q = pca.components_.T[:, :q]

        clusternumber = min([q + self.diff_n_features, X.shape[1]])

        kmeans = KMeans(n_clusters=clusternumber).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

        return X[:, self.indices_]

    def transform(self, X):
        return X[:, self.indices_]


def feature_importance(dataloader,datasets,entities):
    data_dict = load_data(dataloader, datasets, entities)

    fi_dict = dict()
    for d in datasets:
        fi_dict[d] = dict()
        for i, entity in enumerate(entities[d]):
            X = data_dict[d][entity]["data"]
            #y = dataloader.label
            #pfa = PFA(diff_n_features=1, explained_var=0.95)
            #pfa.fit_transform(X)
            #featurekeys = pfa.indices_
            #print(f"{d} - {entity}: {featurekeys}")
            length, channels = X.shape
            is_normal_dist = True
            normal_dist = np.random.normal(0, 1, length)

            for c1 in range(channels):
                if kstest(X[:,c1], normal_dist).pvalue > 0.05:
                    print(c1)
                    is_normal_dist = False

            corr_matrix = np.zeros([channels, channels])
            for c1 in range(channels):
                for c2 in range(channels):
                    corr = pearsonr(X[:, c1], X[:, c2]) if is_normal_dist else spearmanr(X[:, c1], X[:, c2])
                    corr_matrix[c1,c2] = corr.statistic
            #print(corr_matrix)
            #print(d, entity, normal_dist)
            plt.imshow(np.nan_to_num(corr_matrix, nan=1.0), cmap='hot', interpolation='nearest')
            plt.show()


def feature_correlation(dataloader,datasets,entities):
    data_dict = load_data(dataloader, datasets, entities)
    corr_dict = dict()
    for d in datasets:
        corr_dict[d] = dict()
        for i, entity in enumerate(entities[d]):
            filename = f"output/feature_correlations/correlation_matrix_{d}_{entity}.csv"
            if os.path.isfile(filename):
                correlation_matrix = pd.read_csv(filename, header=None)
            else:
                df = pd.DataFrame(data_dict[d][entity]['data'])
                correlation_matrix = df.corr(method='pearson')  # 'spearman' or 'kendall' can also be used
                transformed_data = []
                for i in range(correlation_matrix.shape[0]):
                    for j in range(correlation_matrix.shape[1]):
                        transformed_data.append([j, i, df.iat[i, j]])

                # Create a new DataFrame from the transformed data
                transformed_df = pd.DataFrame(transformed_data, columns=['x', 'y', 'value'])

                # Save the transformed DataFrame to CSV without index and header
                transformed_df.to_csv(filename, index=False, header=False)
                #correlation_matrix.to_csv(filename, index=False, header=False)

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
            plt.title(f"{d} - {entity}: Feature Correlation Heatmap")
            plt.show()