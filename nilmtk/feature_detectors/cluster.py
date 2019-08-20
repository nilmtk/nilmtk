import numpy as np
import pandas as pd

def cluster(X, max_num_clusters=3, exact_num_clusters=None):
    '''Applies clustering on reduced data,
    i.e. data where power is greater than threshold.

    Parameters
    ----------
    X : pd.Series or single-column pd.DataFrame
    max_num_clusters : int

    Returns
    -------
    centroids : ndarray of int32s
        Power in different states of an appliance, sorted
    '''
    # Find where power consumption is greater than 10
    data = _transform_data(X)

    # Find clusters
    centroids = _apply_clustering(data, max_num_clusters, exact_num_clusters)
    centroids = np.append(centroids, 0)  # add 'off' state
    centroids = np.round(centroids).astype(np.int32)
    centroids = np.unique(centroids)  # np.unique also sorts
    # TODO: Merge similar clusters
    return centroids


def _transform_data(data):
    '''Subsamples if needed and converts to column vector (which is what
    scikit-learn requires).

    Parameters
    ----------
    data : pd.Series or single column pd.DataFrame

    Returns
    -------
    data_above_thresh : ndarray
        column vector
    '''

    MAX_NUMBER_OF_SAMPLES = 2000
    MIN_NUMBER_OF_SAMPLES = 20
    DATA_THRESHOLD = 10

    data_above_thresh = data[data > DATA_THRESHOLD].dropna().values
    n_samples = len(data_above_thresh)
    if n_samples < MIN_NUMBER_OF_SAMPLES:
        return np.zeros((MAX_NUMBER_OF_SAMPLES, 1))
    elif n_samples > MAX_NUMBER_OF_SAMPLES:
        # Randomly subsample (we don't want to smoothly downsample
        # because that is likely to change the values)
        random_indices = np.random.randint(0, n_samples, MAX_NUMBER_OF_SAMPLES)
        resampled = data_above_thresh[random_indices]
        return resampled.reshape(MAX_NUMBER_OF_SAMPLES, 1)
    else:
        return data_above_thresh.reshape(n_samples, 1)


def _apply_clustering_n_clusters(X, n_clusters):
    """
    :param X: ndarray
    :param n_clusters: exact number of clusters to use
    :return:
    """
    from sklearn.cluster import KMeans
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)
    return k_means.labels_, k_means.cluster_centers_


def _apply_clustering(X, max_num_clusters, exact_num_clusters=None):
    '''
    Parameters
    ----------
    X : ndarray
    max_num_clusters : int

    Returns
    -------
    centroids : list of numbers
        List of power in different states of an appliance
    '''
    # If we import sklearn at the top of the file then it makes autodoc fail

    from sklearn import metrics

    # Finds whether 2 or 3 gives better Silhouellete coefficient
    # Whichever is higher serves as the number of clusters for that
    # appliance
    num_clus = -1
    sh = -1
    k_means_labels = {}
    k_means_cluster_centers = {}
    k_means_labels_unique = {}

    # If the exact number of clusters are specified, then use that
    if exact_num_clusters is not None:
        labels, centers = _apply_clustering_n_clusters(X, exact_num_clusters)
        return centers.flatten()

    # Exact number of clusters are not specified, use the cluster validity measures
    # to find the optimal number
    for n_clusters in range(1, max_num_clusters):

        try:
            labels, centers = _apply_clustering_n_clusters(X, n_clusters)
            k_means_labels[n_clusters] = labels
            k_means_cluster_centers[n_clusters] = centers
            k_means_labels_unique[n_clusters] = np.unique(labels)
            try:
                sh_n = metrics.silhouette_score(
                    X, k_means_labels[n_clusters], metric='euclidean')

                if sh_n > sh:
                    sh = sh_n
                    num_clus = n_clusters
            except Exception:
                num_clus = n_clusters
        except Exception:
            if num_clus > -1:
                return k_means_cluster_centers[num_clus]
            else:
                return np.array([0])

    return k_means_cluster_centers[num_clus].flatten()


def hart85_means_shift_cluster(pair_buffer_df, columns):

    from sklearn.cluster import MeanShift
    # Creating feature vector
    cluster_df = pd.DataFrame()
    power_types = [col[1] for col in columns]

    if 'active' in power_types:
        cluster_df['active'] = pd.Series(
            pair_buffer_df.apply(
                lambda row: (
                    (np.fabs(
                        row['T1 Active']) +
                        np.fabs(
                        row['T2 Active'])) /
                    2),
                axis=1),
            index=pair_buffer_df.index)
    if 'reactive' in power_types:
        cluster_df['reactive'] = pd.Series(
            pair_buffer_df.apply(
                lambda row: (
                    (np.fabs(
                        row['T1 Reactive']) +
                        np.fabs(
                        row['T2 Reactive'])) /
                    2),
                axis=1),
            index=pair_buffer_df.index)
    if 'apparent' in power_types:
        cluster_df['apparent'] = pd.Series(
            pair_buffer_df.apply(
                lambda row: (
                    (np.fabs(
                        row['T1 Apparent']) +
                        np.fabs(
                        row['T2 Apparent'])) /
                    2),
                axis=1),
            index=pair_buffer_df.index)
    X = cluster_df.values.reshape((len(cluster_df.index), len(columns)))
    ms = MeanShift(bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    return pd.DataFrame(cluster_centers, columns=columns)
