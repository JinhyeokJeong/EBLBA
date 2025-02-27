
import numpy as np
from scipy.stats import norm 
import scipy.integrate as integrate

def simulate_LBA(list_v, list_B, A=0.5, t0=0, s=0, rng=None):
    """ Linear Ballistic Accumulator Model.
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
        - f: pdf of LBA accumulator(s)
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
        - F: cdf of LBA accumulator(s)
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
        - dpdf: defective pdf of i-th (determined by ref) LBA accumulator
    """

    list_v = np.asarray(list_v)
    v_ref = list_v[ref]
    v_rest = np.delete(list_v, ref)

    # TODO: test if I can use a list of B values (i.e., different thresholds across accumulators)
    # need generalization to N alternative situation
    p_ref = pdf_LBA_accumulator(t=t,v=v_ref,b=b,A=A,s=s) # f_{ref(t)}
    p_rest = [(1-cdf_LBA_accumulator(t=t,v=v,b=b,A=A,s=s)) for v in v_rest] #\prod_{j\neq i}{1-F_j}
    p_rest = np.prod(np.vstack(p_rest),axis=0)

    dpdf = p_ref * p_rest # pdf_LBA_accumulator(t=t,v=v_ref,b=b,A=A,s=s)*(1-cdf_LBA_accumulator(t=t,v=v_rest,b=b,A=A,s=s))

    return(dpdf)

def dcdf_from_dpdf(t, dpdf_values):
    """ Approximate the defective CDF values from the discrete defective pdf values. Assumes that x values are wide enough to capture the entire shape of pdf.    
    Note that the last value of dcdf would be equal to the choice probability of choosing the corresponding choice.
    Arguments:
        - t: array of x-values 
        - dpdf_values: array of defective PDF values 
    Returns:
        - dcdf_values: array of corresponding defective CDF values
    """

    dx = np.diff(t)[0] # assume that differences between time points are uniform
    dcdf = np.cumsum(dpdf_values * dx)

    return dcdf 

def cdf_from_pdf(x_values, pdf_values):
    """ Approximate the CDF values from the discrete pdf values. Assumes that x values are wide enough to capture the entire shape of pdf.    
    Arguments:
        - x_values: array of x-values 
        - pdf_values: array of PDF values 
    Returns:
        - cdf_values: array of corresponding CDF values
    """

    # ensure x values are sorted 
    idx_sorted = np.argsort(x_values)
    x_sorted = np.array(x_values)[idx_sorted]
    pdf_sorted = np.array(pdf_values)[idx_sorted]

    # compute the cumulative values for integral (using trapezoidal rule)
    dx = np.diff(x_sorted)
    cdf_values = np.cumsum(pdf_sorted[:-1]*dx) 
    cdf_values /= cdf_values[-1]

    cdf_values = np.instert(cdf_values, 0, 0)

    return cdf_values 

# TODO: make a function to generate predictions of LBA given analytical solutions
# def LBA(list_v, list_B, A=0.5, t0=0, s=0, rng=None):
#     """ Linear Ballistic Accumulator Model.
#     This function simulates a single LBA process (choice and RT) given the parameters.
#     Note that this function does not use a closed form solution. 
#     Arguments:
#         - list_v: drift rate (len(list_v)=number of choice alternatives)
#         - list_B: decision thresholds (positive values)
#         - A: upper bound of uniform distribution of starting point (k ~ U[0,A])
#         - t0: non-decision time
#         - s: between-trial noise of drift rate (SD of Gaussian distribution)
#         - rng: random number generator from numpy.random module
#     Returns:
#         - (choice, rt)
#     """
