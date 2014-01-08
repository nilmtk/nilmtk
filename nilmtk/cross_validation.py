"""
The :mod:`nilmtk.cross_validation` module includes utilities for cross-
validation and performance evaluation. It is based on scikit-learn's 
cross-validation 
APIs(https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cross_validation.py)
"""

from __future__ import print_function, division
import copy


def train_test_split(building, **options):
    """
    Splits a Building object into train/test based on the index of
    building.utility.electricity.mains

    Parameters
    ----------
    building : nilmtk.building
        building must have mains and some submetered information

    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    Returns
    -------
    splitting : list of buildings, where size of this list is 2

    """

    # Finding the buildings utility.electric.mains
    mains = building.utility.electric.mains

    # Since, it is already assumed that all the mains have same DateTime
    # Index, we choose the DateTime index from the first key
    datetime_index = mains[mains.keys()[0]].index

    # Number of samples in the datetime_index
    n_samples = len(datetime_index)
    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)

    if test_size is None and train_size is None:
        test_size = 0.25
        train_size = 0.75
    elif test_size is None:
        test_size = 1 - train_size
    else:
        train_size = 1 - test_size

    # Finding the timestamp which separates the train and the test sets
    split_timestamp = datetime_index[int(n_samples * train_size)]

    # Creating deep copies of building to preserve other information already
    # present in the building
    train, test = copy.deepcopy(building), copy.deepcopy(building)

    # Splitting mains
    for main in building.utility.electric.mains:
        train.utility.electric.mains[
            main] = train.utility.electric.mains[main][:split_timestamp]
        test.utility.electric.mains[
            main] = test.utility.electric.mains[main][split_timestamp:]

    # Splitting appliances
    for appliance in building.utility.electric.appliances:
        train.utility.electric.appliances[
            appliance] = train.utility.electric.appliances[appliance][:split_timestamp]
        test.utility.electric.appliances[
            appliance] = test.utility.electric.appliances[appliance][split_timestamp:]

    return [train, test]
