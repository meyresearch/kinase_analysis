import numpy as np
from scipy.sparse import issparse
from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from deeptime.markov.tools import estimation as msmest
from deeptime.util.types import ensure_dtraj_list
import scipy

class PriorTransitionCountEstimator(TransitionCountEstimator):
    r""" Computes a count matrix based on a counting mode, some discrete trajectories, a lagtime, and
    whether to use sparse matrices. Allows for adding prior counts.

    Parameters
    ----------
    count_mode : str
        The counting mode to use. One of "sample", "sliding", "sliding-effective", and "effective".
        See :meth:`__init__` for a more detailed description.
    dtrajs : array_like or list of array_like
        Discrete trajectories, i.e., a list of arrays which contain non-negative integer values. A single ndarray
        can also be passed, which is then treated as if it was a list with that one ndarray in it.
    lagtime : int
        Distance between two frames in the discretized trajectories under which their potential change of state
        is considered a transition.
    sparse : bool, default=False
        Whether to use sparse matrices or dense matrices. Sparse matrices can make sense when dealing with a lot of
        states.
    n_jobs : int, optional, default=None
        This only has an effect in effective counting. Determines the number of cores to use for estimating
        statistical inefficiencies. Default resolves to number of available cores.
    prior : float, optional, default=0
        The optional prior count to add to the computed count matrix. 

    Returns
    -------
    count_matrix : (N, N) ndarray or sparse array
        The computed count matrix with prior counts added (if specified). Can be ndarray or sparse depending on whether
        sparse was set to true or false. N is the number of encountered states, i.e., :code:`np.max(dtrajs)+1`.

    Example
    -------
    >>> dtrajs = [np.array([0,0,1,1]), np.array([0,0,1])]
    >>> count_matrix = TransitionCountEstimator.count(
    ...     count_mode="sliding", dtrajs=dtrajs, lagtime=1, sparse=False
    ... )
    >>> np.testing.assert_equal(count_matrix, np.array([[2, 2], [0, 1]]))
    """

    def __init__(self, lagtime: int, count_mode: str, n_states=None, sparse=False, prior: float = 0):   
        super().__init__(lagtime=lagtime, count_mode=count_mode, n_states=n_states, sparse=sparse)
        self.prior = prior


    def fit(self, data, *args, **kw):
        r""" Counts transitions at given lag time according to configuration of the estimator.

        Parameters
        ----------
        data : array_like or list of array_like
            discretized trajectories
        """
        from deeptime.markov import count_states
        dtrajs = ensure_dtraj_list(data)

        # basic count statistics
        histogram = count_states(dtrajs, ignore_negative=True)

        # Compute count matrix
        count_mode = self.count_mode
        lagtime = self.lagtime
        count_matrix = PriorTransitionCountEstimator.count(count_mode, dtrajs, lagtime, sparse=self.sparse,
                                                      n_jobs=kw.pop('n_jobs', None), prior=self.prior)
        if self.n_states is not None and self.n_states > count_matrix.shape[0]:
            histogram = np.pad(histogram, pad_width=[(0, self.n_states - count_matrix.shape[0])])
            if issparse(count_matrix):
                count_matrix = scipy.sparse.csr_matrix((count_matrix.data, count_matrix.indices, count_matrix.indptr),
                                                       shape=(self.n_states, self.n_states))
            else:
                n_pad = self.n_states - count_matrix.shape[0]
                count_matrix = np.pad(count_matrix, pad_width=[(0, n_pad), (0, n_pad)])

        # initially state symbols, full count matrix, and full histogram can be left None because they coincide
        # with the input arguments
        self._model = TransitionCountModel(
            count_matrix=count_matrix, counting_mode=count_mode, lagtime=lagtime, state_histogram=histogram
        )
        return self
    

    @staticmethod
    def count(count_mode: str, dtrajs: list[np.ndarray], lagtime: int, sparse: bool = False, n_jobs=None, prior: float = 0):
        """
        Extended count method with support for prior counts.
        """
        # Compute the count matrix using the original method logic
        if count_mode == 'sliding' or count_mode == 'sliding-effective':
            count_matrix = msmest.count_matrix(dtrajs, lagtime, sliding=True, sparse_return=sparse)
            if count_mode == 'sliding-effective':
                count_matrix /= lagtime
        elif count_mode == 'sample':
            count_matrix = msmest.count_matrix(dtrajs, lagtime, sliding=False, sparse_return=sparse)
        elif count_mode == 'effective':
            count_matrix = msmest.effective_count_matrix(dtrajs, lagtime, n_jobs=n_jobs)
            if not sparse and issparse(count_matrix):
                count_matrix = count_matrix.toarray()
        else:
            raise ValueError(f"Count mode {count_mode} is unknown.")
        
        # Add prior counts if specified
        if prior > 0:
            if issparse(count_matrix):
                count_matrix.data += prior 
            else:
                count_matrix += prior

        return count_matrix
