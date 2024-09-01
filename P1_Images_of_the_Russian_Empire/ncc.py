# The following code is retrieved from
# https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html, Aimo Winkelmann
# Date retrieved: 24. aug. 2024

import numpy as np


def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    # return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data - mean_data) / (std_data)


def ncc2(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0 / (data0.size - 1)) * np.sum(norm_data(data0) * norm_data(data1))
