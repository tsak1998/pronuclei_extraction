from datetime import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from models import GrowthEventsEnum, Distribution, DistributionParams

THEORETICAL_DISTRIBUTIONS: list[str] = [
    'norm', 'weibull_min', 'gumbel_r', 'uniform', 'expon'
]

possible_params: list[str] = ['shape', 'loc', 'scale']


class DistributionFitter:
    """
    Class to find the best fitted to the given data
    """

    def __init__(self) -> None:
        pass

    def fit_distribution(self, dist_name: str,
                         data: np.ndarray) -> Distribution:

        scipy_dist = getattr(stats, dist_name)
        fitted_parameters: NamedTuple = scipy_dist.fit(data)

        fitted_distribution = scipy_dist(*fitted_parameters)

        dist_params = DistributionParams(**{
            param: p
            for p, param in zip(fitted_parameters, possible_params)
        })
        return Distribution(distribution=dist_name,
                            params=dist_params,
                            fitted_dist=fitted_distribution)

    def find_best_distribution(self, data: np.ndarray) -> Distribution:
        """
        Finds the best fitted distribution(with the Kolmogorov-Smirnov test)
        among the provided theoretical distributions.

        Params:
        data: (array): The data to be fitted
        """

        params = {}
        dist_results = []
        for dist_name in THEORETICAL_DISTRIBUTIONS:
            scipy_dist = getattr(stats, dist_name)
            param = scipy_dist.fit(data)
            params[dist_name] = param

            # Applying the Kolmogorov-Smirnov test
            test_statistic, p_value = stats.kstest(data, dist_name, args=param)
            # reject the null hypothesis
            if p_value < 0.05:
                continue
            dist_results.append((dist_name, test_statistic))

        # select the best fitted distribution
        best_dist, _ = (max(dist_results, key=lambda item: item[1]))

        scipy_dist = getattr(stats, best_dist)
        best_fitted_distribution = fit_distribution(dist_name=best_dist,
                                                    data=data)

        return best_fitted_distribution
