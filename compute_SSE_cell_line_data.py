# Author: Xiuxia Du
# April 2021

# This code is to compute the Within-Cluster Sum of Squares (SSE) and Between-Cluster Sum of Squares (SSB) for k-means, hierarchical, and DBSCAN.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN

import Machining_learning_class.config as config

in_file_name = 'SCLC_study_output_filtered_2.csv'
in_file_full_name = config.data_folder + in_file_name
data_in = pd.read_csv(in_file_full_name, index_col=0, header=0)

num_of_samples = data_in.shape[0]
num_of_variables = data_in.shape[1]

num_of_clusters = 2
overall_mean = data_in.mean(axis=0)

clustering_methods = ['kmeans', 'single', 'complete', 'average', 'centroid', 'ward', 'DBSCAN']
clustering_results = pd.DataFrame(np.zeros((num_of_samples, len(clustering_methods))), \
                                  index=data_in.index, columns=clustering_methods)

# ------------------------------------------------------------
# do kmeans
# ------------------------------------------------------------
object_kmeans = KMeans(n_clusters=num_of_clusters, random_state=0)
object_kmeans.fit(data_in)
cluster_centers = object_kmeans.cluster_centers_
cluster_labels = object_kmeans.labels_
cluster_labels_series = pd.Series(cluster_labels, index=data_in.index)

clustering_results.loc[:, 'kmeans'] = cluster_labels_series

# compute SSE and SSB
# Note: SSE is also available as inertia_ in object_kmeans
sse = pd.Series(np.zeros(len(clustering_methods)), index=clustering_methods)
ssb = sse.copy()

for i in range(num_of_clusters):
    II = (np.where(cluster_labels == i))[0]
    num_of_samples = len(II)

    ssb['kmeans'] = ssb['kmeans'] + num_of_samples * np.linalg.norm(cluster_centers[i, :] - overall_mean) * \
          np.linalg.norm(cluster_centers[i, :] - overall_mean)

    for j in range(num_of_samples):
        current_sample = data_in.iloc[II[j], :]
        sse['kmeans'] = sse['kmeans'] + np.linalg.norm(current_sample - cluster_centers[i, :]) * \
              np.linalg.norm(current_sample - cluster_centers[i, :])

# ------------------------------------------------------------
# do hierarchical
# ------------------------------------------------------------
all_linkages = ['single', 'complete', 'average', 'centroid', 'ward']

for cur_linkage in all_linkages:

    Z = hierarchy.linkage(data_in, cur_linkage)
    cluster_labels = hierarchy.fcluster(Z, t=num_of_clusters, criterion='maxclust')
    cluster_labels_series = pd.Series(cluster_labels, index=data_in.index)

    clustering_results.loc[:, cur_linkage] = cluster_labels_series

    fig, ax = plt.subplots()
    dn = hierarchy.dendrogram(Z, labels=data_in.index.values.tolist(), orientation='right')
    ax.set_title('Hierarchial: ' + cur_linkage)
    fig.show()
    out_file_name = 'dendrogram-for-' + cur_linkage + 'linkage.pdf'
    out_file_full_name = config.result_folder + out_file_name
    fig.savefig(out_file_full_name)

    # compute SSE and SSB
    cluster_centers = np.zeros((num_of_clusters, num_of_variables))
    unique_cluster_labels = np.unique(cluster_labels)

    for i in range(len(unique_cluster_labels)):
        current_cluster_label = unique_cluster_labels[i]
        II = np.where(cluster_labels == current_cluster_label)
        II = np.asarray(II)
        II = II[0]
        cluster_centers[i, :] = data_in.iloc[II, :].mean(axis=0)

    # compute SSE
    for i in range(num_of_clusters):
        II = (np.where(cluster_labels == i))[0]
        num_of_samples = len(II)

        ssb[cur_linkage] = ssb[cur_linkage] + num_of_samples * np.linalg.norm(cluster_centers[i, :] - overall_mean) * \
              np.linalg.norm(cluster_centers[i, :] - overall_mean)

        for j in range(num_of_samples):
            current_sample = data_in.iloc[II[j], :]
            sse[cur_linkage] = sse[cur_linkage] + np.linalg.norm(current_sample - cluster_centers[i, :]) * \
                  np.linalg.norm(current_sample - cluster_centers[i, :])

# ------------------------------------------------------------
# do DBSCAN
# ------------------------------------------------------------
radius_list = np.arange(2E5, 3E6, 2E5).tolist()
min_samples = 5

for radius in radius_list:
    object_DBSCAN = DBSCAN(eps=radius, min_samples=min_samples)
    object_DBSCAN.fit(data_in)
    cluster_labels = object_DBSCAN.labels_
    print(cluster_labels)
    clustering_results.loc[:, 'DBSCAN'] = cluster_labels

radius = 1E6
object_DBSCAN = DBSCAN(eps=radius, min_samples=min_samples)
object_DBSCAN.fit(data_in)
cluster_labels = object_DBSCAN.labels_
clustering_results.loc[:, 'DBSCAN'] = cluster_labels

# ------------------------------------------------------------
# overall plot
# ------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(range(len(clustering_methods)), sse, color='blue', marker='*', label='SSE')
ax.plot(range(len(clustering_methods)), ssb, color='red', marker='*', label='SSB')
ax.legend()
ax.set_xticks(range(len(clustering_methods)))
ax.set_xticklabels(clustering_methods, rotation=90)
ax.set_ylabel('SSE and SSB')
ax.set_title('SSE and SSB for different clustering methods')
fig.show()

out_file_name = 'clustering_results.csv'
out_file_full_name = config.result_folder + out_file_name
clustering_results.to_csv(out_file_full_name)

out_file_name = 'sse-ssb-clustering-methods.pdf'
out_file_full_name = config.result_folder + out_file_name
fig.savefig(out_file_full_name)

# ------------------------------------------------------------
# determining k in kmeans
# ------------------------------------------------------------
k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sse_ssb_kmeans = pd.DataFrame(np.zeros((len(k_list), 2)), index=k_list, columns=['SSE', 'SSB'])

inertia_kmeans = pd.Series(np.zeros(len(k_list)), index=k_list)

for cur_k in k_list:
    object_kmeans = KMeans(n_clusters=cur_k, random_state=1)
    object_kmeans.fit(data_in)
    cluster_centers = object_kmeans.cluster_centers_
    cluster_labels = object_kmeans.labels_

    inertia_kmeans[cur_k] = object_kmeans.inertia_

    for i in range(cur_k):
        II = (np.where(cluster_labels == i))[0]
        num_of_samples = len(II)

        sse_ssb_kmeans.loc[cur_k, 'SSB'] = sse_ssb_kmeans.loc[cur_k, 'SSB'] + \
                                           num_of_samples * np.linalg.norm(cluster_centers[i, :] - overall_mean) * \
                                           np.linalg.norm(cluster_centers[i, :] - overall_mean)

        for j in range(num_of_samples):
            current_sample = data_in.iloc[II[j], :]
            sse_ssb_kmeans.loc[cur_k, 'SSE'] = sse_ssb_kmeans.loc[cur_k, 'SSE'] + \
                                               np.linalg.norm(current_sample - cluster_centers[i, :]) * \
                                               np.linalg.norm(current_sample - cluster_centers[i, :])

fig, ax = plt.subplots()
ax.plot(k_list, sse_ssb_kmeans.loc[:, 'SSE'], marker='*', color='blue', label='SSE')
ax.plot(k_list, sse_ssb_kmeans.loc[:, 'SSB'], marker='*', color='red', label='SSB')
ax.legend()
ax.set_xlabel('k')
ax.set_title('SSE and SSB for different values of k in kmeans')
fig.show()
out_file_name = 'sse-ssb-kmeans.pdf'
out_file_full_name = config.result_folder + out_file_name
fig.savefig(out_file_full_name)

fig, ax = plt.subplots()
ax.plot(k_list, inertia_kmeans, marker='*', color='blue', label='SSE')
ax.plot(k_list, sse_ssb_kmeans.loc[:, 'SSB'], marker='*', color='red', label='SSB')
ax.legend()
ax.set_xlabel('k')
ax.set_title('SSE and SSB for different values of k in kmeans')
fig.show()

out_file_name = 'SSE_SSB_kmeans.csv'
out_file_full_name = config.result_folder + out_file_name
sse_ssb_kmeans.to_csv(out_file_full_name)

xx = 1