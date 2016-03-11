

train.shape

tr1 = train[:500]
tr2 = train[501:1000]

type(tr1)
tr1.values.shape
tr2.values.shape

import scipy as sp
from scipy.signal import correlate
import scipy as sp
from scipy.stats import pearsonr
import numpy as np
xxx = correlate(tr1.values, tr2.values)
xxx.shape

X, Y = tr1.values, tr2.values
X.shape
Y.shape


# iter(range(X.shape[0]))
from tomorrow import threads
from util.decorator import get_func_time

@get_func_time
def make_combinations(X, Y):
    n1 = X.shape[0]
    n2 = Y.shape[0]

    _combinations = []
    _cmat = np.zeros((n1,n2))
    i = 0
    while (i < n1):
        j = 0
        while (j < n2):
            # _cmat[i,j] = sum(pearsonr(X[i], Y[j]))/2
            _combinations.append([i,j])
            j += 1
        i += 1
    print "Done generate combinations"
    return iter(_combinations)

# tmp = make_combinations(X, Y)

@threads(50)
def cor(x,y):
    return sum(pearsonr(x, y))/2

@get_func_time
def cross_correlation(X, Y):
    _comb = make_combinations(X,Y)
    print "Start calculation"
    res = [cor(X[_c[0]],Y[_c[1]]) for _c in _comb]
    print res[:10]
    print "Done calculation"
    return "Done"

_res = cross_correlation(X,Y)






