from __future__ import print_function, division
import numpy as np
import pandas as pd 
import sys


# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


def find_steady_states(dataframe, min_n_samples=2, stateThreshold=15,
                       noise_level=70):
    """Finds steady states given a datafram of power

    Parameters
    ----------

    dataframe: pd.DataFrame with DateTimeIndex
    min_n_samples(int): number of samples to consider constituting a
             steady state.
    stateThreshold: maximum difference between highest and lowest 
        value in steady state.
    noise_level: the level used to define significant 
        appliances, transitions below this level will be ignored. 
        See Hart 1985. p27.


    Returns
    -------

    """


      # Tells whether we have both real and reactive power or only real power
    num_measurements = len(dataframe.columns)
    estimatedSteadyPower = np.array([0] * num_measurements)
    lastSteadyPower = np.array([0] * num_measurements)
    previousMeasurement = np.array([0] * num_measurements)

    # These flags store state of power

    instantaneousChange = False  # power changing this second
    ongoingChange = False  # power change in progress over multiple seconds

    index_transitions = []  # Indices to use in returned Dataframe
    index_steadystates = []
    transitions = []  # holds information on transitions
    steadyStates = []  # steadyStates to store in returned Dataframe
    N = 0  # N stores the number of samples in state
    time = dataframe.iloc[0].name  # first state starts at beginning

    # Iterate over the rows performing algorithm
    print ("Finding Edges, please wait ...", end="\n")
    sys.stdout.flush()

    for row in dataframe.itertuples():

        # test if either active or reactive moved more than threshold
        # http://stackoverflow.com/questions/17418108/elegant-way-to-perform-tuple-arithmetic
        # http://stackoverflow.com/questions/13168943/expression-for-elements-greater-than-x-and-less-than-y-in-python-all-in-one-ret

        # Step 2: this does the threshold test and then we sum the boolean
        # array.
        thisMeasurement = row[1:3]
        # logging.debug('The current measurement is: %s' % (thisMeasurement,))
        # logging.debug('The previous measurement is: %s' %
        # (previousMeasurement,))

        stateChange = np.fabs(
            np.subtract(thisMeasurement, previousMeasurement))
        # logging.debug('The State Change is: %s' % (stateChange,))

        if np.sum(stateChange > stateThreshold):
            instantaneousChange = True
        else:
            instantaneousChange = False

        # Step 3: Identify if transition is just starting, if so, process it
        if (instantaneousChange and (not ongoingChange)):

            # Calculate transition size
            lastTransition = np.subtract(estimatedSteadyPower, lastSteadyPower)
            # logging.debug('The steady state transition is: %s' %
            # (lastTransition,))

            # Sum Boolean array to verify if transition is above noise level
            if np.sum(np.fabs(lastTransition) > noise_level):
                # 3A, C: if so add the index of the transition start and the
                # power information

                # Avoid outputting first transition from zero
                index_transitions.append(time)
                # logging.debug('The current row time is: %s' % (time))
                transitions.append(lastTransition)

                # I think we want this, though not specifically in Hart's algo notes
                # We don't want to append a steady state if it's less than min samples in length.
                # if N > min_n_samples:
                index_steadystates.append(time)
                # logging.debug('The ''time'' stored is: %s' % (time))
                # last states steady power
                steadyStates.append(estimatedSteadyPower)

            # 3B
            lastSteadyPower = estimatedSteadyPower
            # 3C
            time = row[0]

        # Step 4: if a new steady state is starting, zero counter
        if instantaneousChange:
            N = 0

        # Hart step 5: update our estimate for steady state's energy
        estimatedSteadyPower = np.divide(
            np.add(np.multiply(N, estimatedSteadyPower),
                   thisMeasurement), (N + 1))
        # logging.debug('The steady power estimate is: %s' %
        #    (estimatedSteadyPower,))
        # Step 6: increment counter
        N = N + 1

        # Step 7
        ongoingChange = instantaneousChange

        # Step 8
        previousMeasurement = thisMeasurement

    print("Edge detection complete.")

    print("Creating transition frame ...")
    sys.stdout.flush()

    cols_transition = {1: ['active transition'],
                       2: ['active transition', 'reactive transition']}

    cols_steady = {1: ['active average'],
                   2: ['active average', 'reactive average']}

    
    if len(index_transitions)==0:
        # No events
        return pd.DataFrame(), pd.DataFrame()
    else:
        transitions = pd.DataFrame(data=transitions, index=index_transitions,
                                   columns=cols_transition[num_measurements])
        print("Transition frame created.")

        print("Creating states frame ...")
        sys.stdout.flush()
        steadyStates = pd.DataFrame(data=steadyStates, index=index_steadystates,
                                    columns=cols_steady[num_measurements])
        print("States frame created.")
        print("Finished.")
        return steadyStates, transitions


def cluster(X, max_num_clusters=3):
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
    centroids = _apply_clustering(data, max_num_clusters)
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


def _apply_clustering(X, max_num_clusters):
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
    from sklearn.cluster import KMeans
    from sklearn import metrics

    # sklearn produces lots of DepreciationWarnings with PyTables
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Finds whether 2 or 3 gives better Silhouellete coefficient
    # Whichever is higher serves as the number of clusters for that
    # appliance
    num_clus = -1
    sh = -1
    k_means_labels = {}
    k_means_cluster_centers = {}
    k_means_labels_unique = {}
    for n_clusters in range(1, max_num_clusters):

        try:
            k_means = KMeans(init='k-means++', n_clusters=n_clusters)
            k_means.fit(X)
            k_means_labels[n_clusters] = k_means.labels_
            k_means_cluster_centers[n_clusters] = k_means.cluster_centers_
            k_means_labels_unique[n_clusters] = np.unique(k_means_labels)
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
