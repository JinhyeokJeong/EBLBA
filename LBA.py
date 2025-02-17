
import numpy as np
from scipy.stats import norm 

def LBA(list_v, list_B, A=0.5, t0=0, s=0, rng=None):
    """ Linear Ballistic Model.
    This function simulates a single LBA process (choice and RT) given the parameters.
    Note that this function does not use a closed form solution. 
    Arguments:
        - list_v: drift rate (len(list_v)=number of choice alternatives)
        - list_B: decision thresholds (positive values)
        - A: upper bound of uniform distribution of starting point (k ~ U[0,A])
        - t0: non-decision time
        - s: between-trial noise of drift rate (SD of Gaussian distribution)
        - rng: random number generator from numpy.random module
    Returns:
        - (choice, rt)
    """

    if rng is None:
        rng = np.random.default_rng()

    list_v += rng.normal(loc=0, scale=s, size=len(list_v))    

    # assumed that len(list_evd) == len(list_B)
    if len(list_B)==1:
        list_B = [list_B for _ in range(len(list_v))]

    list_k = rng.uniform(0, A + np.finfo(float).eps) # [low,high] = [low,high+eps)    

    RTs = np.array([((list_B[i]-list_k)/list_v[i])+t0 for i in range(len(list_v))])
    # RTs = np.array([((b-k)/evd)+t0 for evd in list_evd]))

    choice = np.argmin(RTs)
    RT = RTs[choice]
    # rt = ((b - k) / d1) + t0

    return choice, RT

def pdf_LBA_accumulator(t,v, b, A, s):
    """ PDF for the time taken for LBA accumualtors to reach threshold
    Arguments:
        - t: time variable
        - v: drift rate (could be either a single value or an array)
        - b: decision threshold (could be either a single value or an array)
        - A: upper bound of uniform distribution of starting point (single value)
        - s: between-trial noise of drift rate (single value)
    Returns: 
    """

    f = (1/A)*( -v*norm.cdf((b-A-t*v)/(t*s)) + s*norm.pdf((b-A-t*v)/(t*s)) + v*norm.cdf((b-t*v)/(t*s)) - s*norm.pdf((b-t*v)/(t*s)) )

    return f

def cdf_LBA_accumulator(t,v, b, A, s):
    """ CDF for the time taken for LBA accumualtors to reach threshold
    Arguments:
        - t: time variable
        - v: drift rate (could be either a single value or an array)
        - b: decision threshold (could be either a single value or an array)
        - A: upper bound of uniform distribution of starting point (single value)
        - s: between-trial noise of drift rate (single value)
    Returns: 
    """
    F = 1 + ((b-A-t*v)/A)*norm.cdf((b-A-t*v)/(t*s)) - ((b-t*v)/A)*norm.cdf((b-t*v)/(t*s)) + \
        ((t*s)/A)*norm.pdf((b-A-t*v)/(t*s)) - ((t*s)/A)*norm.pdf((b-t*v)/(t*s))
    return F

def defective_pdf_LBA(t,list_v,b,A,s, ref=0):
    """ defective PDF of response times for the LBA accumulators.
    Arguments:
        - t: time variable
        - list_v: a list (or array) of drift rates. Need at least two accumulators (two drift rates)
        - b: decision threshold (could be either a single value or an array)
        - A: upper bound of uniform distribution of starting point (single value)
        - s: between-trial noise of drift rate (single value)
    Returns:
    """

    list_v = np.asarray(list_v)
    v_ref = list_v[ref]
    v_rest = np.delete(v_ref, ref)

    dPDF = pdf_LBA_accumulator(t,v_ref,s,b,A)*(1-cdf_LBA_accumulator(t,v_rest,s,b,A))

    return(dPDF)