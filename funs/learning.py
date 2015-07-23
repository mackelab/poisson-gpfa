import util

import numpy as np
import scipy as sp
import scipy.optimize as op
# import numdifftools as nd
import pdb

from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.numdiff import approx_hess

import matplotlib.pyplot as plt

# ==================================================================================
# For batch EM
# ==================================================================================

# for C, d

def MStepObservationCost(vecCd, xdim, ydim, experiment, infRes):
    '''
    Function written Macke, Buesing, Sahani 2015 for the PLDS model.
    The Observation parameter cost function is identical to the P-GPFA model.
    Translated to Python by Hooram Nam 2015
    '''
    T = experiment.T
    numTrials = len(experiment.data)
    C,d = util.vecCdtoCd(vecCd, xdim ,ydim)

    CC = np.zeros([ydim,xdim**2])

    for yd in range(ydim):
        CC[yd,:] = np.reshape(np.outer(C[yd,:],C[yd,:]),xdim**2)

    f = 0
    df = np.zeros(np.shape(C))
    dfd = np.zeros(ydim)

    for trial in range(numTrials):
        y = experiment.data[trial]['Y']
        m = infRes['post_mean'][trial]
        vsm = np.reshape(infRes['post_vsm'][trial],[T,xdim**2])

        hh = (np.dot(C,m) + np.asarray([d]).T)
        rho = (np.dot(CC, vsm.T))
        yhat = np.exp(hh+rho/2)
        f = f + sum(sum(y*hh-yhat))

    return -f/numTrials

def MStepObservationCost_grad(vecCd, xdim, ydim, experiment, infRes):
    '''
    Function written Macke, Buesing, Sahani 2015 for the PLDS model.
    The Observation parameter cost function is identical to the P-GPFA model.
    Translated to Python by Hooram Nam 2015
    '''
    T = experiment.T
    numTrials = len(experiment.data)
    C, d = util.vecCdtoCd(vecCd, xdim, ydim)

    CC = np.zeros([ydim,xdim**2])

    for yd in range(ydim):
        CC[yd,:] = np.reshape(np.outer(C[yd,:],C[yd,:]),xdim**2)

    dfC = np.zeros(np.shape(C))
    dfd = np.zeros(ydim)

    for trial in range(numTrials):
        y = experiment.data[trial]['Y']
        m = infRes['post_mean'][trial]
        vsm = np.reshape(infRes['post_vsm'][trial],[T,xdim**2])

        hh = np.float64(np.dot(C,m) + np.asarray([d]).T)
        rho = np.float64(np.dot(CC, vsm.T))

        yhat = np.exp(hh+rho/2)

        vecC = np.reshape(C.T,xdim*ydim)

        T1 = np.reshape(np.dot(yhat,vsm).T, [xdim, xdim*ydim]).T
        T2 = (T1 * np.asarray([vecC]).T)
        T3 = np.reshape(T2.T,(xdim,xdim,ydim))
        TT = np.sum(T3,1)

        dfC = dfC + np.dot(y-yhat,m.T) - TT.T
        dfd = dfd + np.sum(y-yhat,1)
    
    vecdf = -util.CdtoVecCd(dfC,dfd)/numTrials

    return vecdf

def learnLTparams(oldParams, infRes, experiment, CdOptimMethod, CdMaxIter=None, verbose = False):
    
    [ydim,xdim] = np.shape(oldParams['C'])
    vecCd = util.CdtoVecCd(oldParams['C'], oldParams['d'])
    xinit = np.zeros(xdim*ydim+ydim)
    xinit = vecCd

    if False:
        invPriorCov = np.diag(np.zeros(xdim*ydim+ydim))
        invPriorCov = np.diag(np.ones(xdim*ydim+ydim))
        def cost_prior(vecCd): 
            cost = MStepObservationCostWithPrior(
                vecCd, oldParams, xdim, ydim, experiment, infRes,invPriorCov)
            return cost
        def cost_prior_grad(vecCd): 
            grad = MStepObservationCostWithPrior_grad(
                vecCd, oldParams, xdim, ydim, experiment, infRes, invPriorCov)
            return grad

        obsGradCheck = op.check_grad(MStepObservationCost, MStepObservationCost_grad, vecCd, xdim, ydim, experiment, infRes)
        print('observation grad check = ' + str(obsGradCheck))
        apprxGrad = op.approx_fprime(xinit,MStepObservationCost,1e-8*np.ones(xdim*ydim+ydim), xdim, ydim, experiment, infRes)
        calcdGrad = MStepObservationCost_grad(xinit,xdim, ydim, experiment, infRes)
        plt.plot(apprxGrad,linewidth = 10, color = 'k', alpha = 0.4)
        plt.plot(calcdGrad,linewidth = 2, color = 'k', alpha = 0.4)
        plt.legend(['approximated','calculated'])
        plt.title('Approx. vs. calculated Grad of C,d learning cost')
        plt.tight_layout()
        plt.show()
        pdb.set_trace()
    
    resCd = op.minimize(
        fun = MStepObservationCost,
        x0 = xinit,
        args = (xdim, ydim, experiment, infRes),
        jac = MStepObservationCost_grad,
        method = CdOptimMethod,
        options = {'disp': verbose, 'maxiter':CdMaxIter})

    if resCd.success==True & verbose:
        print('Cd optimization successful.')
    if resCd.success==False & verbose: 
        print('Cd optimization unsuccessful.')
        
    newvecCd = resCd.x
    costFun = resCd.fun

    newC, newd = util.vecCdtoCd(newvecCd, xdim, ydim)
    return newC, newd, costFun

# for tau

def makePrecomp(infRes):
    '''
    Function written by Byron & Yu 2009 for the GPFA model. 
    The timescale constant cost function is exactly the same as the P-GPFA case. 
    Translated to Python by Hooram Nam 2015
    '''
    xdim, T = np.shape(infRes['post_mean'][0])
    numTrials = len(infRes['post_mean'])

    Ttile1=np.tile(np.asarray([np.arange(T)]).T,[1,T])+1
    Ttile2=np.tile(np.asarray([np.arange(T)]),[T,1])+1
    Tdif = Ttile1-Ttile2
    difSq = Tdif*Tdif

    precomp = []
    PautoSum = []

    for xd in range(xdim):
        tempPautoSum = np.zeros([T,T])
        for tr in range(numTrials):
            tempPautoSum = tempPautoSum + infRes['post_vsmGP'][tr][:,:,xd] + np.outer(infRes['post_mean'][tr][xd,:],infRes['post_mean'][tr][xd,:])

        precomp.append({
            'T':T, 
            'Tdif':Tdif, 
            'difSq':difSq, 
            'numTrials':numTrials,
            'PautoSum':tempPautoSum})
    return precomp

def MStepGPtimescaleCost(p, precomp, epsNoise):
    '''
    Function written by Byron & Yu 2009 for the GPFA model. 
    The timescale constant cost function is exactly the same in the P-GPFA case. 
    Translated to Python by Hooram Nam 2015
    '''
    T = precomp['T']

    temp = (1-epsNoise)*np.exp(-np.exp(p)/2*precomp['difSq'])
    K = temp + epsNoise*np.eye(T)
    dKdgamma = -0.5*temp*precomp['difSq']
    
    dEdgamma = 0
    f = 0

    Thalf = int(np.ceil(T/2))
    Kinv = np.linalg.inv(K)
    sign, logdet = np.linalg.slogdet(K)
    logdet_K = sign*logdet

    KinvM = np.dot(Kinv[:Thalf,:],dKdgamma)
    KinvMKinv = np.dot(KinvM,Kinv)
    
    dg_KinvM = np.diag(KinvM)
    tr_KinvM = 2*sum(dg_KinvM) - T%2*dg_KinvM[-1]

    mkr = int(np.ceil(0.5*T**2))

    KinvVec = np.reshape(Kinv,T*T)
    PSallVec = np.reshape(precomp['PautoSum'],T*T)
    PS1vec = PSallVec[:mkr]
    PS2vec = PSallVec[mkr:]
    PS2vec = PS2vec[::-1]

    KinvMKinvVec = np.reshape(KinvMKinv, mkr)
    
    dEdgamma = -0.5*precomp['numTrials']*tr_KinvM + 0.5*np.dot(PS1vec,KinvMKinvVec)+ 0.5*np.dot(PS2vec,KinvMKinvVec)
    f = -0.5* precomp['numTrials']*logdet_K - 0.5*np.dot(PSallVec,KinvVec)

    return -f

def MStepGPtimescaleCost_grad(p, precomp, epsNoise):
    '''
    Function written by Byron, Yu in 2009 for the GPFA model. 
    The timescale constant cost function is exactly the same in the P-GPFA case. 
    Translated to Python by Hooram Nam 2015
    '''
    T = precomp['T']
    
    temp = (1-epsNoise)*np.exp(-np.exp(p)/2*precomp['difSq'])
    K = temp + epsNoise*np.eye(T)
    dKdgamma = -0.5*temp*precomp['difSq']
    
    dEdgamma = 0
    f = 0

    Thalf = int(np.ceil(T/2))
    Kinv = np.linalg.inv(K)
    sign, logdet = np.linalg.slogdet(K)
    logdet_K = sign*logdet

    KinvM = np.dot(Kinv[:Thalf,:],dKdgamma)
    KinvMKinv = np.dot(KinvM,Kinv)
    
    dg_KinvM = np.diag(KinvM)
    tr_KinvM = 2*sum(dg_KinvM) - T%2*dg_KinvM[-1]

    mkr = int(np.floor(0.5*T**2))

    KinvVec = np.reshape(Kinv,T*T)
    PSallVec = np.reshape(precomp['PautoSum'],T*T)
    PS1vec = PSallVec[:mkr]
    PS2vec = PSallVec[mkr:]
    PS2vec = PS2vec[::-1]

    KinvMKinvVec = np.reshape(KinvMKinv, mkr)
    
    dEdgamma = -0.5*precomp['numTrials']*tr_KinvM + 0.5*np.dot(PS1vec,KinvMKinvVec) + 0.5*np.dot(PS2vec,KinvMKinvVec)
    f = -0.5* precomp['numTrials']*logdet_K - 0.5*np.dot(PSallVec,KinvVec)

    return -dEdgamma*np.exp(p)

def learnGPparams(oldParams, infRes, experiment):
    xdim, T = np.shape(infRes['post_mean'][0])
    binSize = experiment.binSize
    oldTau = oldParams['tau']*1000/binSize
    
    precomp = makePrecomp(infRes)
    
    tempTau = np.zeros(xdim)

    pOptimizeDetails = [[]]*xdim
    for xd in range(xdim): 
        initp = np.log(1/oldTau[xd]**2)

        if False: # gradient check
            gradcheck = op.check_grad(MStepGPtimescaleCost,MStepGPtimescaleCost_grad,initp,precomp[0],0.001)
            print('tau learning grad check = ' + str(gradcheck))
            apprxGrad = op.approx_fprime(initp,MStepGPtimescaleCost,1e-8,precomp[xd],0.001)
            calcdGrad = MStepGPtimescaleCost_grad(initp,precomp[xd],0.001)
            plt.plot(apprxGrad,linewidth = 10, color = 'k', alpha = 0.4)
            plt.plot(calcdGrad,linewidth = 2, color = 'k', alpha = 0.4)
            plt.legend(['approximated','calculated'])
            plt.title('Approx. vs. calculated Grad of Tau learning cost')
            plt.tight_layout()
            plt.show()
            pdb.set_trace()

        res = op.minimize(
            fun = MStepGPtimescaleCost,
            x0 = initp,
            args = (precomp[xd], 0.001),
            jac = MStepGPtimescaleCost_grad,
            options = {'disp': False,'gtol':1e-8})
        pOptimizeDetails[xd] = res
        tempTau[xd] = (1/np.exp(res.x))**(0.5)

    newTau = tempTau*binSize/1000
    return newTau, pOptimizeDetails

def updateParams(oldParams, infRes, experiment, CdOptimMethod='BFGS', CdMaxIter=None, tauMaxIter=None, verbose=False):
    if verbose: print('Learning C,d...')
    newC, newd, obsOptimDetails = learnLTparams(
        oldParams = oldParams, 
        infRes = infRes, 
        experiment = experiment, 
        CdOptimMethod = CdOptimMethod, 
        CdMaxIter = CdMaxIter,
        verbose = verbose)    
    
    if verbose: print('Learning GP timescale constants')
    newTau, dynOptimDetails = learnGPparams(oldParams, infRes, experiment)
    newParams = {'C':newC, 'd':newd, 'tau':newTau}
    optimDetails = {'Cd':obsOptimDetails,'tau':dynOptimDetails}
    return newParams, optimDetails


# ==================================================================================
# For online EM 
# ==================================================================================

# for C, d

def update_d_closedForm(oldParams, infRes, experiment):
    T = experiment.T
    numTrials = len(experiment.data)
    ydim, xdim = np.shape(oldParams['C'])

    spikes = np.zeros([experiment.ydim, T*numTrials])
    for tr in range(numTrials):
        spikes[:,tr*T:(tr+1)*T] = experiment.data[tr]['Y']
    meanY = np.mean(spikes,1)
    d = np.log(meanY*T)
    
    C = oldParams['C']

    CC = np.zeros([ydim,xdim**2])

    for yd in range(ydim):
        CC[yd,:] = np.reshape(np.outer(C[yd,:],C[yd,:]),xdim**2)

    yhat = np.zeros([ydim,T])
    for trial in range(numTrials):
        y = experiment.data[trial]['Y']
        m = infRes['post_mean'][trial]
        vsm = np.reshape(infRes['post_vsm'][trial],[T,xdim**2])

        hh = (np.dot(C,m))
        rho = (np.dot(CC, vsm.T))
        yhat = yhat + np.exp(hh+rho/2)

    d = d - np.log(np.mean(yhat,1))
    #=====
    T1 = np.log(meanY+1e-5) # for numerical stability add a small number before taking the log
    T2 = np.zeros(ydim)
    for trial in range(numTrials):
        yhat = 0
        for time in range(T):
            yhat = yhat + np.exp(C.dot(infRes['post_mean'][trial][:,time]) + 1/2*np.diag(np.dot(C,infRes['post_vsm'][trial][time]).dot(C.T)))
        T2 = T2 + yhat
    d = T1 - np.log(T2)/(T*numTrials)
    # ddebug = d
    # pdb.set_trace()
    return d

def MStep_C_costWithPrior(vecC, oldParams, xdim, ydim, experiment, infRes, invPriorCov):
    '''
    Original function MStepGPtimescaleCost written for MATLAB by Byron & Yu 2009.
    Adds Gaussian prior terms for the elements of C and d. Their prior variances are given by the
    variables sigma_C and sigma_d. The prior means are given by the previous estimate of C and d.
    This effectively amounts to adding a L2 regularizer term. 
    Added regularizer term by Hooram Nam 2015
    '''
    T = experiment.T
    numTrials = len(experiment.data)
    C = np.reshape(vecC,np.shape(oldParams['C']))

    oldC = oldParams['C']
    oldd = oldParams['d']
    oldVecC = oldC.flatten()

    CC = np.zeros([ydim,xdim**2])
    for yd in range(ydim):
        CC[yd,:] = np.reshape(np.outer(C[yd,:],C[yd,:]),xdim**2)

    f = 0

    for trial in range(numTrials):
        y = experiment.data[trial]['Y']
        m = infRes['post_mean'][trial]
        vsm = np.reshape(infRes['post_vsm'][trial],[T,xdim**2])

        hh = (np.dot(C,m) + np.asarray([oldd]).T)
        rho = (np.dot(CC, vsm.T))
        yhat = np.exp(hh+rho/2)

        f = f + sum(sum(y*hh-yhat))

    if invPriorCov != None:
        priorTerm = 1/2*np.dot(np.dot(vecC-oldVecC,invPriorCov),vecC-oldVecC)
        f = f + priorTerm
    # pdb.set_trace()
    return -f

def MStep_C_costWithPrior_grad(vecC, oldParams, xdim, ydim, experiment, infRes, invPriorCov):
    '''
    Original function MStepGPtimescaleCost_cost written for MATLAB by Byron & Yu 2009.
    Adds Gaussian prior terms for the elements of C and d. Their prior variances are given by the
    variables sigma_C and sigma_d. The prior means are given by the previous estimate of C and d.
    This effectively amounts to adding a L2 regularizer term. 
    Added regularizer term by Hooram Nam 2015
    '''
    T = experiment.T
    numTrials = len(experiment.data)
    C = np.reshape(vecC, np.shape(oldParams['C']))
    
    oldC = oldParams['C']
    oldd = oldParams['d']
    oldVecC = oldC.flatten()

    CC = np.zeros([ydim,xdim**2])
    for yd in range(ydim):
        CC[yd,:] = np.reshape(np.outer(C[yd,:],C[yd,:]),xdim**2)

    dfC = np.zeros(np.shape(C))

    for trial in range(numTrials):
        y = experiment.data[trial]['Y']
        m = infRes['post_mean'][trial]
        vsm = np.reshape(infRes['post_vsm'][trial],[T,xdim**2])

        hh = (np.dot(C,m) + np.asarray([oldd]).T)
        rho = (np.dot(CC, vsm.T))
        yhat = np.exp(hh+rho/2)

        T1 = np.reshape(np.dot(yhat,vsm).T, [xdim, xdim*ydim]).T
        T2 = (T1 * np.asarray([np.reshape(C.T,xdim*ydim)]).T)
        T3 = np.reshape(T2.T,(xdim,xdim,ydim))
        TT = np.sum(T3,1)

        dfC = dfC + np.dot(y-yhat,m.T) - TT.T
    
    if invPriorCov == None:
        vecdf = -dfC.flatten()
    else:
        dPriorTerm = np.dot(invPriorCov,vecC-oldVecC)
        vecdf = -dfC.flatten() - dPriorTerm
    return vecdf


def MStepObservationCostWithPrior(vecCd, 
    oldParams, xdim, ydim, experiment, infRes, invPriorCov):
    '''
    Original function MStepGPtimescaleCost written for MATLAB by Byron & Yu 2009.
    Adds Gaussian prior terms for the elements of C and d. Their prior variances are given by the
    variables sigma_C and sigma_d. The prior means are given by the previous estimate of C and d.
    This effectively amounts to adding a L2 regularizer term. 
    Added regularizer term by Hooram Nam 2015
    '''
    T = experiment.T
    numTrials = len(experiment.data)
    C,d = util.vecCdtoCd(vecCd, xdim ,ydim)

    oldC = oldParams['C']
    oldd = oldParams['d']
    oldVecCd = util.CdtoVecCd(oldC,oldd)

    CC = np.zeros([ydim,xdim**2])

    for yd in range(ydim):
        CC[yd,:] = np.reshape(np.outer(C[yd,:],C[yd,:]),xdim**2)

    f = 0
    df = np.zeros(np.shape(C))
    dfd = np.zeros(ydim)

    for trial in range(numTrials):
        y = experiment.data[trial]['Y']
        m = infRes['post_mean'][trial]
        vsm = np.reshape(infRes['post_vsm'][trial],[T,xdim**2])

        hh = (np.dot(C,m) + np.asarray([d]).T)
        rho = (np.dot(CC, vsm.T))
        yhat = np.exp(hh+rho/2)

        f = f + sum(sum(y*hh-yhat))

    priorTerm = 1/2*np.dot(np.dot(vecCd-oldVecCd,invPriorCov),vecCd-oldVecCd)
    # pdb.set_trace()
    # print(f/(f+priorTerm))
    # print(priorTerm/(f+priorTerm))
    return -f/numTrials - priorTerm

def MStepObservationCostWithPrior_grad(vecCd, 
    oldParams, xdim, ydim, experiment, infRes, invPriorCov):
    '''
    Original function MStepGPtimescaleCost_cost written for MATLAB by Byron & Yu 2009.
    Adds Gaussian prior terms for the elements of C and d. Their prior variances are given by the
    variables sigma_C and sigma_d. The prior means are given by the previous estimate of C and d.
    This effectively amounts to adding a L2 regularizer term. 
    Added regularizer term by Hooram Nam 2015
    '''
    T = experiment.T
    numTrials = len(experiment.data)
    C,d = util.vecCdtoCd(vecCd, xdim, ydim)
    
    oldC = oldParams['C']
    oldd = oldParams['d']
    oldVecCd = util.CdtoVecCd(oldC,oldd)

    CC = np.zeros([ydim,xdim**2])
    for yd in range(ydim):
        CC[yd,:] = np.reshape(np.outer(C[yd,:],C[yd,:]),xdim**2)

    dfC = np.zeros(np.shape(C))
    dfd = np.zeros(ydim)

    for trial in range(numTrials):
        y = experiment.data[trial]['Y']
        m = infRes['post_mean'][trial]
        vsm = np.reshape(infRes['post_vsm'][trial],[T,xdim**2])

        hh = np.float64(np.dot(C,m) + np.asarray([d]).T)
        rho = np.float64(np.dot(CC, vsm.T))
        yhat = np.exp(hh+rho/2)

        vecC = np.reshape(C.T,xdim*ydim)

        T1 = np.reshape(np.dot(yhat,vsm).T, [xdim, xdim*ydim]).T
        T2 = (T1 * np.asarray([vecC]).T)
        T3 = np.reshape(T2.T,(xdim,xdim,ydim))
        TT = np.sum(T3,1)

        dfC = dfC + np.dot(y-yhat,m.T) - TT.T
        dfd = dfd + np.sum(y-yhat,1)
    
    dPriorTerm = np.dot(invPriorCov,vecCd-oldVecCd)

    vecdf = -util.CdtoVecCd(dfC,dfd)/numTrials - dPriorTerm
    return vecdf

def learnLTparamsWithPrior(oldParams, infRes, experiment, 
    CdOptimMethod, regularizer_stepsize_Cd, prevInvPriorCov, 
    covOpts='useDiag', updateCdJointly=True, hessTol = 1e-5, verbose = False):  
    [ydim,xdim] = np.shape(oldParams['C'])

    if updateCdJointly == True:
        vecCd = util.CdtoVecCd(oldParams['C'], oldParams['d'])
        xinit = np.zeros(xdim*ydim+ydim)
        xinit = vecCd
        if covOpts == 'useHessian':
            hessCd_naive = util.approx_jacobian(
                vecCd,MStepObservationCostWithPrior_grad, hessTol, 
                oldParams, xdim, ydim, experiment, infRes, prevInvPriorCov)
            invPriorCov = -hessCd_naive

            # pdb.set_trace()
            # plt.plot(np.diag(invPriorCov),'g')

            # def cost_prior_grad(vecCd): 
            #     cost = MStepObservationCostWithPrior_grad(
            #         vecCd, oldParams, xdim, ydim, experiment, infRes, prevInvPriorCov)
            #     return cost
            # invPriorCov = -approx_fprime(vecCd, cost_prior_grad, centered=True)
            # plt.plot(np.diag(invPriorCov),'b')
            # pdb.set_trace()

            # def cost_prior(vecCd): 
            #     cost = MStepObservationCostWithPrior(
            #         vecCd, oldParams, xdim, ydim, experiment, infRes,prevInvPriorCov)
            #     return cost
            # invPriorCov = -approx_hess(vecCd, cost_prior_grad, hessTol)
            # plt.plot(np.diag(invPriorCov),'r')
            # pdb.set_trace()

            # def cost_prior_grad(vecCd): 
            #     grad = MStepObservationCostWithPrior_grad(
            #         vecCd, oldParams, xdim, ydim, experiment, infRes, invPriorCov)
            #     return grad
            # hess = nd.Jacobian(cost_prior_grad)
            # invPriorCov = -hess(vecCd)
            # plt.plot(np.diag(invPriorCov),'r')
            # # pdb.set_trace()
            # invPriorCov = -prevInvPriorCov

        if covOpts == 'useDiag':
            invPriorCov = -np.diag(np.ones(xdim*ydim+ydim))/(regularizer_stepsize_Cd**2)

        if False: # bench for grad check
            def cost_prior(vecCd): 
                cost = MStepObservationCostWithPrior(
                    vecCd, oldParams, xdim, ydim, experiment, infRes,invPriorCov)
                return cost
            def cost_prior_grad(vecCd): 
                grad = MStepObservationCostWithPrior_grad(
                    vecCd, oldParams, xdim, ydim, experiment, infRes, invPriorCov)
                return grad
            approxGrad = op.approx_fprime(
                vecCd, MStepObservationCostWithPrior,1e-6, 
                oldParams, xdim, ydim, experiment, infRes,invPriorCov)
            trueGrad = cost_prior_grad(vecCd)
            # plt.plot(approxGrad,'g')
            # plt.plot(trueGrad,'r')
            tol = 1e-3
            print(np.abs(trueGrad-approxGrad) < tol)
            op.check_grad(cost_prior,cost_prior_grad,np.random.rand(xdim*ydim+ydim))
            # hhh = util.hessian(vecCd, cost_prior_grad, tol)
            # plt.plot(np.diag(invPriorCov))
            # pdb.set_trace()

        if CdOptimMethod == 'L-BFGS-B':
            bounds = [(None,None)]*(xdim*ydim+ydim)
            resCd = op.minimize(
                fun = MStepObservationCostWithPrior,
                x0 = xinit,
                args = (oldParams, xdim, ydim, experiment, infRes, invPriorCov),
                jac = MStepObservationCostWithPrior_grad,
                bounds = bounds,
                method = CdOptimMethod,
                options = {'disp': verbose, 'gtol':1e-10})        
        if CdOptimMethod != 'L-BFGS-B':
            resCd = op.minimize(
                fun = MStepObservationCostWithPrior,
                x0 = xinit,
                args = (oldParams, xdim, ydim, experiment, infRes, invPriorCov),
                jac = MStepObservationCostWithPrior_grad,
                method = CdOptimMethod,
                options = {'disp': verbose, 'gtol':1e-10})
        if resCd.success==True & verbose==True:
            print('Cd optimization successful.')
        if resCd.success==False & verbose==True: 
            print('Cd optimization unsuccessful.')

        # debug line
        if False: pdb.set_trace()
        newvecCd = resCd.x
        costFun = resCd.fun
        newC, newd = util.vecCdtoCd(newvecCd, xdim, ydim)

    if updateCdJointly == False:
        d_closedForm = update_d_closedForm(oldParams, infRes, experiment)
        oldParams['d'] = d_closedForm
        vecC = oldParams['C'].flatten()
        xinit = vecC

        if covOpts == 'useHessian':
            hessCd_naive = util.approx_jacobian(
                vecC,MStep_C_costWithPrior_grad, hessTol, 
                oldParams, xdim, ydim, experiment, infRes, prevInvPriorCov[:xdim*ydim,:xdim*ydim])
            invPriorCov = -hessCd_naive
            # invPriorCov = -prevInvPriorCov
        if covOpts == 'useDiag':
            invPriorCov = -(np.diag(np.ones(xdim*ydim))/(regularizer_stepsize_Cd**2))

        if False: # bench for grad check
            def cost_prior(vecC): 
                cost = MStep_C_costWithPrior(
                    vecC, oldParams, xdim, ydim, experiment, infRes, invPriorCov)
                return cost
            def cost_prior_grad(vecC): 
                grad = MStep_C_costWithPrior_grad(
                    vecC, oldParams, xdim, ydim, experiment, infRes, invPriorCov)
                return grad
            approxGrad = op.approx_fprime(
                vecC, MStep_C_costWithPrior, 1e-3, 
                oldParams, xdim, ydim, experiment, infRes,invPriorCov)
            trueGrad = cost_prior_grad(vecC)
            tol = 1e-3
            pdb.set_trace()
            print(np.abs(trueGrad-approxGrad) < tol)
        
        resC = op.minimize(
            fun = MStep_C_costWithPrior,
            x0 = xinit,
            args = (oldParams, xdim, ydim, experiment, infRes, invPriorCov),
            jac = MStep_C_costWithPrior_grad,
            method = CdOptimMethod,
            options = {'disp': verbose, 'gtol':1e-10})
        newd = d_closedForm
        newC = np.reshape(resC.x,np.shape(oldParams['C']))
        costFun = resC.fun
    return newC, newd, costFun, invPriorCov


# for tau

def MStepGPtimescaleCostWithPrior(p, precomp, epsNoise, binSize, oldTau, regularizer_stepsize_tau):
    '''
    Original function MStepGPtimescaleCost written for MATLAB by Byron & Yu 2009 for the GPFA model. 
    The timescale constant cost function is exactly the same in the P-GPFA case. 
    Translated to Python by Hooram Nam 2015
    Added regularizer term by Hooram Nam 2015
    '''
    tau = binSize/1000*(1/np.exp(p))**(1/2)
    regularizer = 1/2*(tau - oldTau)**2/(regularizer_stepsize_tau**2)

    T = precomp['T']

    temp = (1-epsNoise)*np.exp(-np.exp(p)/2*precomp['difSq'])
    K = temp + epsNoise*np.eye(T)
    dKdgamma = -0.5*temp*precomp['difSq']
    
    dEdgamma = 0
    f = 0

    Thalf = int(np.ceil(T/2))
    Kinv = np.linalg.inv(K)
    sign, logdet = np.linalg.slogdet(K)
    logdet_K = sign*logdet

    KinvM = np.dot(Kinv[:Thalf,:],dKdgamma)
    KinvMKinv = np.dot(KinvM,Kinv)
    
    dg_KinvM = np.diag(KinvM)
    tr_KinvM = 2*sum(dg_KinvM) - T%2*dg_KinvM[-1]

    mkr = int(np.ceil(0.5*T**2))

    KinvVec = np.reshape(Kinv,T*T)
    PSallVec = np.reshape(precomp['PautoSum'],T*T)
    PS1vec = PSallVec[:mkr]
    PS2vec = PSallVec[mkr:]
    PS2vec = PS2vec[::-1]

    KinvMKinvVec = np.reshape(KinvMKinv, mkr)
    
    dEdgamma = -0.5*precomp['numTrials']*tr_KinvM + 0.5*np.dot(PS1vec,KinvMKinvVec)+ 0.5*np.dot(PS2vec,KinvMKinvVec)
    f = -0.5* precomp['numTrials']*logdet_K - 0.5*np.dot(PSallVec,KinvVec)

    return -f + regularizer

def MStepGPtimescaleCostWithPrior_grad(p, precomp, epsNoise, binSize, oldTau, regularizer_stepsize_tau):
    '''
    Original function MStepGPtimescaleCost written for MATLAB by Byron & Yu 2009 for the GPFA model. 
    The timescale constant cost function is exactly the same in the P-GPFA case. 
    Translated to Python by Hooram Nam 2015
    Added regularizer term by Hooram Nam 2015
    '''
    tau = binSize/1000*(1/np.exp(p))**(1/2)
    df_regularizer = (tau - oldTau)/(regularizer_stepsize_tau**2)

    T = precomp['T']
    
    temp = (1-epsNoise)*np.exp(-np.exp(p)/2*precomp['difSq'])
    K = temp + epsNoise*np.eye(T)
    dKdgamma = -0.5*temp*precomp['difSq']
    
    dEdgamma = 0
    f = 0

    Thalf = int(np.ceil(T/2))
    Kinv = np.linalg.inv(K)
    sign, logdet = np.linalg.slogdet(K)
    logdet_K = sign*logdet

    KinvM = np.dot(Kinv[:Thalf,:],dKdgamma)
    KinvMKinv = np.dot(KinvM,Kinv)
    
    dg_KinvM = np.diag(KinvM)
    tr_KinvM = 2*sum(dg_KinvM) - T%2*dg_KinvM[-1]

    mkr = int(np.floor(0.5*T**2))

    KinvVec = np.reshape(Kinv,T*T)
    PSallVec = np.reshape(precomp['PautoSum'],T*T)
    PS1vec = PSallVec[:mkr]
    PS2vec = PSallVec[mkr:]
    PS2vec = PS2vec[::-1]

    KinvMKinvVec = np.reshape(KinvMKinv, mkr)
    
    dEdgamma = -0.5*precomp['numTrials']*tr_KinvM + 0.5*np.dot(PS1vec,KinvMKinvVec) + 0.5*np.dot(PS2vec,KinvMKinvVec)
    f = -0.5* precomp['numTrials']*logdet_K - 0.5*np.dot(PSallVec,KinvVec)

    return -dEdgamma*np.exp(p) + df_regularizer

def learnGPparamsWithPrior(oldParams, infRes, experiment, tauOptimMethod, regularizer_stepsize_tau):
    xdim, T = np.shape(infRes['post_mean'][0])
    binSize = experiment.binSize
    oldTau = oldParams['tau']*1000/binSize
    
    precomp = makePrecomp(infRes)
    
    tempTau = np.zeros(xdim)

    pOptimizeDetails = [[]]*xdim
    for xd in range(xdim): 
        initp = np.log(1/oldTau[xd]**2)

        if False: # gradient check and stuff
            gradcheck = op.check_grad(
                MStepGPtimescaleCostWithPrior,
                MStepGPtimescaleCostWithPrior_grad,
                initp,precomp[0],0.001,binSize, oldParams['tau'][xd], regularizer_stepsize_tau)
            print('tau learning grad check = ' + str(gradcheck))
            pdb.set_trace()
            apprxGrad = op.approx_fprime(
                initp,MStepGPtimescaleCostWithPrior,1e-8,
                precomp[xd],0.001,binSize,oldParams['tau'][xd],regularizer_stepsize_tau)
            calcdGrad = MStepGPtimescaleCostWithPrior_grad(
                initp,precomp[xd],0.001,binSize,oldParams['tau'][xd],regularizer_stepsize_tau)
            plt.plot(apprxGrad,linewidth = 10, color = 'k', alpha = 0.4)
            plt.plot(calcdGrad,linewidth = 2, color = 'k', alpha = 0.4)
            plt.legend(['approximated','calculated'])
            plt.title('Approx. vs. calculated Grad of Tau learning cost')
            plt.tight_layout()
            plt.show()
            def cost(p): 
                cost = MStepGPtimescaleCostWithPrior(
                    p, precomp[xd], 0.001, binSize, oldParams['tau'][xd], regularizer_stepsize_tau)
                return cost
            def cost_grad(p): 
                grad = MStepGPtimescaleCostWithPrior_grad(
                    p, precomp[xd], 0.001, binSize, oldParams['tau'][xd], regularizer_stepsize_tau)
                return grad
            pdb.set_trace()

        if False: # bench for setting hessian as inverse variance
            hessTau = op.approx_fprime([initp], MStepGPtimescaleCost_grad, 1e-14, 
                precomp[xd], 0.001)
            priorVar = -1/hessTau
            regularizer_stepsize_tau = np.sqrt(np.abs(priorVar))
            # pdb.set_trace()

        res = op.minimize(
            fun = MStepGPtimescaleCostWithPrior,
            x0 = initp,
            args = (precomp[xd], 0.001, binSize, oldParams['tau'][xd], regularizer_stepsize_tau),
            jac = MStepGPtimescaleCostWithPrior_grad,
            options = {'disp': False,'gtol':1e-10},
            method = tauOptimMethod)
        pOptimizeDetails[xd] = res
        tempTau[xd] = (1/np.exp(res.x))**(0.5)

    newTau = tempTau*binSize/1000
    return newTau, pOptimizeDetails


def updateParamsWithPrior(
    oldParams, infRes, experiment, 
    CdOptimMethod, tauOptimMethod, 
    regularizer_stepsize_Cd, regularizer_stepsize_tau, prevInvPriorCov, 
    covOpts='useHessian', verbose=False, updateCdJointly=True, hessTol = 1e-5):
    '''
    Called by engine.PPGPFAfit. Performs the M-step for 'fullyUpdateWithPrior' as the 
    onlineParamUpdateMethod argument when initializing class engine.PPGPFAfit()
    '''
    if verbose: print('Learning C,d...')
    newC, newd, obsOptimDetails, invPriorCov = learnLTparamsWithPrior(
        oldParams = oldParams, 
        infRes = infRes, 
        experiment = experiment, 
        CdOptimMethod = CdOptimMethod, 
        regularizer_stepsize_Cd = regularizer_stepsize_Cd,
        prevInvPriorCov = prevInvPriorCov,
        covOpts = covOpts,
        updateCdJointly = updateCdJointly,
        hessTol = hessTol,
        verbose = verbose)    

    if verbose: print('Learning GP timescale constants')
    newTau, dynOptimDetails = learnGPparamsWithPrior(
        oldParams = oldParams, 
        infRes = infRes, 
        experiment = experiment,
        tauOptimMethod = tauOptimMethod, 
        regularizer_stepsize_tau = regularizer_stepsize_tau)

    # newTau, dynOptimDetails = learnGPparams(oldParams, infRes, experiment)
    newParams = {'C':newC, 'd':newd, 'tau':newTau}
    optimDetails = {'Cd':obsOptimDetails,'tau':dynOptimDetails}
    return newParams, optimDetails, invPriorCov


# ==================================================================================
# For online EM with gradient descent
# ==================================================================================

# for C, d

def learnLTparamsGradDescent(oldParams, infRes, experiment, stepSize, cumHess, updateCdJointly = True, hessTol = 1e-5):
    [ydim, xdim] = np.shape(oldParams['C'])
    numTrials = len(experiment.data)

    if updateCdJointly == True:
        vecCd = util.CdtoVecCd(oldParams['C'],oldParams['d'])
        def Q(vecCd): return -MStepObservationCost(vecCd, xdim, ydim, experiment, infRes)
        def Q_grad(vecCd): return -MStepObservationCost_grad(vecCd, xdim, ydim, experiment, infRes)
        # hess = nd.Jacobian(Q_grad)
        g = Q_grad(vecCd)
        # h = hess(vecCd)
        h = util.approx_jacobian(vecCd,Q_grad,hessTol)
        # h = h + np.ones(xdim*ydim+ydim)
        # h = hess(vecCd)
        # pdb.set_trace()
        vecCd = vecCd - stepSize*np.dot(np.linalg.inv(h),g)
        newC,newd = util.vecCdtoCd(vecCd,xdim,ydim)

    if updateCdJointly == False:
        d = update_d_closedForm(oldParams, infRes, experiment)
        vecC = oldParams['C'].flatten()
        invPriorCov = np.zeros([len(vecC),len(vecC)]) # don't need the regularizer for gradient descent
        def Q(vecC): return -MStep_C_costWithPrior(vecC, oldParams, xdim, ydim, experiment, infRes, invPriorCov=None)
        def Q_grad(vecC): return -MStep_C_costWithPrior_grad(vecC, oldParams, xdim, ydim, experiment, infRes, invPriorCov=None)
        
        g = Q_grad(vecC)
        h = util.approx_jacobian(vecC,Q_grad,hessTol)
        # pdb.set_trace()
        vecC = vecC - stepSize*np.dot(np.linalg.inv(h),g)
        newC = np.reshape(vecC,np.shape(oldParams['C']))
        newd = d

    return newC, newd, h

def learnGPparamsGradDescent(oldParams, infRes, experiment, stepSize, hessTol):
    [ydim, xdim] = np.shape(oldParams['C'])
    numTrials = len(experiment.data)
    precomp = makePrecomp(infRes)


    for xd in range(xdim):
        def Q(p): MStepGPtimescaleCost(p, precomp[xd], 0.001) 
        def Q_grad(p): MStepGPtimescaleCost_grad(p, precomp[xd], 0.001) 
        



        grad = Q_grad(oldParams['tau'][xd])
        # pdb.set_trace()
        hess = util.approx_jacobian(oldParams['tau'][xd], Q_grad, hessTol)
        
        oldTau = oldParams['tau'][xd]
        newTau = oldTau - stepSize*g*h
        pdb.set_trace()

    return 0

def updateParamsWithGradDescent(oldParams, infRes, experiment, stepSize, cumHess, regularizer_stepsize_tau, tauOptimMethod, 
    updateCdJointly = True, verbose=False, hessTol = 1e-5):
    '''
    Called by engine.PPGPFAfit. Performs the M-step for 'fullyUpdateWithPrior' as the 
    onlineParamUpdateMethod argument when initializing class engine.PPGPFAfit()
    '''
    if verbose: print('Learning C,d...')
    newC, newd, hess = learnLTparamsGradDescent(
        oldParams = oldParams, 
        infRes = infRes, 
        experiment = experiment, 
        stepSize = stepSize, 
        cumHess = cumHess,
        updateCdJointly = updateCdJointly, 
        hessTol = hessTol)
    
    if verbose: print('Learning GP timescale constants')
    # dummy = learnGPparamsGradDescent(
    #     oldParams = oldParams,
    #     infRes = infRes,
    #     experiment = experiment,
    #     stepSize = stepSize,
    #     hessTol = hessTol)

    newTau, dynOptimDetails = learnGPparamsWithPrior(
        oldParams = oldParams, 
        infRes = infRes, 
        experiment = experiment,
        tauOptimMethod = tauOptimMethod, 
        regularizer_stepsize_tau = regularizer_stepsize_tau)

    # newTau, dynOptimDetails = learnGPparams(oldParams, infRes, experiment)
    newParams = {'C':newC, 'd':newd, 'tau':newTau}
    optimDetails = {'Cd':None,'tau':dynOptimDetails}
    return newParams, optimDetails, hess