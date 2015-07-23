import math
import numpy as np
import scipy

import inference
import util
import pdb

def PosteriorMCMC(experiment, params, maxSampleIter, trial):
    [ydim, xdim] = np.shape(params['C'])
    numTrials = len(experiment.data)
    trialDur = experiment.trialDur
    binSize = experiment.binSize
    T = experiment.T
    # make big parameters
    C_big, d_big = util.makeCd_big(params,T)
    K_big, K = util.makeK_big(params, trialDur, binSize)
    K_bigInv = np.linalg.inv(K_big)
    
    x0 = np.ndarray.flatten(np.zeros([xdim*T,1]))

    y = experiment.data[trial]['Y']
    ybar = np.ndarray.flatten(np.reshape(y, ydim*T))

    def log_like_fn(x): return -inference.negLogPosteriorUnNorm(x, ybar, C_big, d_big, K_bigInv, xdim, ydim)

    sampData = []
    cholKBig = np.linalg.cholesky(K_big)

    for i in range(maxSampleIter):
        x0,cur_lnpdf = elliptical_slice(x0, cholKBig, log_like_fn)
        sampData.append(x0)

    # MCMCsampCov = np.cov()
    return np.asarray(sampData)


def elliptical_slice(initial_theta, prior, lnpdf, pdf_params=(),
                     cur_lnpdf=None,angle_range=None):
    """
    NAME:
       elliptical_slice
    PURPOSE:
       Markov chain update for a distribution with a Gaussian "prior" factored out
    INPUT:
       initial_theta - initial vector
       prior - cholesky decomposition of the covariance matrix
               (like what np.linalg.cholesky returns),
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params= parameters to pass to the pdf
       cur_lnpdf= value of lnpdf at initial_theta (optional)
       angle_range= Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    HISTORY:
       Originally written in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
       2012-02-24 - Written - Bovy (IAS)
    """
    D= len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf= lnpdf(initial_theta,*pdf_params)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1: #prior = prior sample
        nu= prior
    else: #prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu= np.dot(prior,np.random.normal(size=D))
    hh = math.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi= np.random.uniform()*2.*math.pi
        phi_min= phi-2.*math.pi
        phi_max= phi
    else:
        # Randomly center bracket on current point
        phi_min= -angle_range*np.random.uniform()
        phi_max= phi_min + angle_range
        phi= np.random.uniform()*(phi_max-phi_min)+phi_min

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        xx_prop = initial_theta*math.cos(phi) + nu*math.sin(phi)
        cur_lnpdf = lnpdf(xx_prop,*pdf_params)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform()*(phi_max - phi_min) + phi_min
    return (xx_prop,cur_lnpdf)
