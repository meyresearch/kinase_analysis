# Functions for the validating MSMs

import deeptime as dt 
from deeptime.markov.msm import BayesianMSM
from deeptime.util.validation import implied_timescales

from typing import *
import numpy as np
from tqdm import tqdm


def its_convergence(dtrajs: List[np.ndarray], lagtimes=[1,10,50,100,200,500,1000], n_samples=10):
    models = []
    for lagtime in tqdm(lagtimes, total=len(lagtimes)):
        models.append(BayesianMSM(n_samples, lagtime=lagtime).fit(dtrajs).fit_fetch(dtrajs))
    its_data = implied_timescales(models)

    return its_data

