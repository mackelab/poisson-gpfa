import pdb
import numpy as np
import scipy as sp
import scipy.optimize as op
import util

import matplotlib.pyplot as plt

import time

# Laplace Inference -----------------------------------------------------------
def negLogPosteriorUnNorm(xbar, ybar, C_big, d_big, K_bigInv, xdim, ydim):
    xbar = np.ndarray.flatten(np.asarray(xbar))
    ybar = np.ndarray.flatten(np.asarray(ybar))
    T = int(len(d_big)/ydim)

    C_big = np.asarray(C_big)
    d_big = np.asarray(d_big)

    K_bigInv = np.asarray(K_bigInv)

    A = np.dot(C_big.T, xbar) + d_big
    Aexp = np.exp(A)

    L1 = np.dot(Aexp, np.ones(ydim*T))
    L2 = - np.dot(ybar, A.T)
    L3 = 0.5*np.dot(xbar,np.dot(K_bigInv,xbar))

    L = L1 + L2 + L3

    # pdb.set_trace()
    return L

def negLogPosteriorUnNorm_grad(xbar, ybar, C_big, d_big, K_bigInv, xdim, ydim):
    xbar = np.asarray(xbar)
    ybar = np.asarray(ybar)

    A = np.dot(C_big.T, xbar) + d_big
    A = np.float64(A)
    Aexp = np.exp(A)

    dL1 = np.dot(Aexp,C_big.T)
    dL2 = - np.dot(ybar, C_big.T)
    dL3 = np.dot(xbar, K_bigInv)

    dL = dL1 + dL2 + dL3

    return dL

def negLogPosteriorUnNorm_hess(xbar, ybar, C_big, d_big, K_bigInv, xdim, ydim):
    xbar = np.asarray(xbar)
    ybar = np.asarray(ybar)

    T = int(len(xbar)/xdim)

    A = np.dot(C_big.T, xbar) + d_big
    A = np.float64(A)

    Aexp = np.exp(A)
    Aexpdiagonal = sp.sparse.spdiags(Aexp,0,ydim*T,ydim*T)
    temp = Aexpdiagonal.dot(C_big.T)

    ddL = np.dot(C_big, temp) + K_bigInv

    return ddL

def laplace(experiment, params, prevOptimRes = None, returnOptimRes = True, verbose = False, optimMethod = 'Newton-CG'):
    '''
    laplaceInfRes, -post_lik = laplace(experiment, params)
    '''
    ridge = 0

    [ydim,T] = np.shape(experiment.data[0]['Y'])
    [ydim, xdim] = np.shape(params['C'])
    numTrials = len(experiment.data)
    trialDur = experiment.trialDur
    binSize = experiment.binSize

    # make big parameters
    C_big, d_big = util.makeCd_big(params,T)
    K_big, K = util.makeK_big(params, trialDur, binSize)
    K_bigInv = np.linalg.inv(K_big)
    
    x_post_mean = []
    x_post_cov = []
    x_vsmGP = []
    x_vsm = []

    post_lik = 0
    
    # store current optimization result to use as initialization for inference in next EM iteration
    lapOptimRes = []

    for trial in range(numTrials):
        if verbose: print('laplace inference trajectory of trial ' +str(trial+1) +'...')
        y = experiment.data[trial]['Y']
        ybar = np.ndarray.flatten(np.reshape(y, ydim*T))

        if prevOptimRes == None:
            xInit = np.ndarray.flatten(np.zeros([xdim*T,1]))
        else:
            xInit = prevOptimRes[trial]

        # Automatic differentiation doesn't work
        if False:
            from ad import gh
            def objective(x): return negLogPosteriorUnNorm(x,ybar,C_big,d_big,K_bigInv,xdim,ydim)
            grad,hess = gh(objective)
            pdb.set_trace()
            resLap = op.minimize(
                fun = objective,
                x0 = xInit,
                method=optimMethod,
                # args = (ybar, C_big, d_big, K_bigInv, xdim, ydim),
                jac = grad,
                hess = hess,
                options = {'disp': False,'maxiter': 10000})

        resLap = op.minimize(
            fun = negLogPosteriorUnNorm,
            x0 = xInit,
            method=optimMethod,
            args = (ybar, C_big, d_big, K_bigInv, xdim, ydim),
            jac = negLogPosteriorUnNorm_grad,
            hess = negLogPosteriorUnNorm_hess,
            options = {'disp': False,'maxiter': 10000})
        lapOptimRes.append(resLap.x)
        post_lik = post_lik + resLap.fun
        x_post_mean.append(np.reshape(resLap.x,[xdim,T]))
        hess = negLogPosteriorUnNorm_hess(resLap.x, ybar, C_big, d_big, K_bigInv, xdim, ydim)
        PostCovGP = np.linalg.inv(hess)
        # PostCovGP = hess

        # resNCG = op.fmin_ncg(
        #     f = negLogPosteriorUnNorm,
        #     x0 = xInit,
        #     fprime = negLogPosteriorUnNorm_grad,
        #     fhess = negLogPosteriorUnNorm_hess,
        #     args = (ybar, C_big, d_big, K_bigInv, xdim, ydim),
        #     disp = False,
        #     full_output = True)
        # lapOptimRes.append(resNCG[0]) 
        # post_lik = post_lik + resNCG[1]
        # x_post_mean.append(np.reshape(resNCG[0],[xdim,T]))
        # hess = -negLogPosteriorUnNorm_hess(resNCG[0], ybar, C_big, d_big, K_bigInv, xdim, ydim)
        # PostCovGP = -np.linalg.inv(hess)

        # resLaplace = op.minimize(
        #     fun = negLogPosteriorUnNorm,
        #     x0 = xInit,
        #     method='TNC',
        #     args = (ybar, C_big, d_big, K_bigInv, xdim, ydim),
        #     jac = negLogPosteriorUnNorm_grad,
        #     hess = negLogPosteriorUnNorm_hess,
        #     options = {'disp': False})
        # post_lik = post_lik + resLaplace.fun
        # x_post_mean.append(np.reshape(resLaplace.x,[xdim,T]))
        # hess = negLogPosteriorUnNorm_hess(resLaplace.x, ybar, C_big, d_big, K_bigInv, xdim, ydim)
        # PostCovGP = -np.linalg.inv(hess)
        
        PostCovGP = PostCovGP + ridge*np.diag(np.ones(xdim*T))
        x_post_cov.append(PostCovGP)

        temp_vsmGP = np.zeros([T,T,xdim])
        for kk in range(xdim):
            temp_vsmGP[:,:,kk] = PostCovGP[kk*T:(kk+1)*T, kk*T:(kk+1)*T]
        x_vsmGP.append(temp_vsmGP)

        temp_vsm = np.zeros([T,xdim,xdim])
        for kk in range(T):
            temp_vsm[kk][:,:] = PostCovGP[kk::T,kk::T]
        x_vsm.append(temp_vsm)
        # pdb.set_trace()

    post_lik = post_lik / numTrials
    laplaceInfRes = {
        'post_mean': x_post_mean,
        'post_cov' : x_post_cov,
        'post_vsm': x_vsm,
        'post_vsmGP': x_vsmGP}

    if returnOptimRes == True:
        return laplaceInfRes, -post_lik, lapOptimRes
    else:
        return laplaceInfRes, -post_lik

# Variational Inference --------------------------------------------------
def VIPostCov(K_bigInv, C_big, lamb):
    postPrecision = K_bigInv + np.dot(np.dot(C_big,np.diag(lamb)), C_big.T)
    postCovariance = np.linalg.inv(postPrecision + 1e-6*np.diag(np.diag(postPrecision))) # slow
    return postCovariance, postPrecision

def VIPostMean(K_big, C_big, y_bar, lamb):
    return -np.dot(np.dot(K_big,C_big), (lamb-y_bar).T)

def dualProblem(lamb, ybar, C_big, K_big, K_bigInv, d_big):
    varMean = VIPostMean(K_big,C_big,ybar,lamb)
    
    postCov, postPrec = VIPostCov(K_bigInv,C_big,lamb)

    lmy = lamb-ybar
    
    A = 1/2*np.dot(lmy.T,np.dot(C_big.T,np.dot(K_big,np.dot(C_big,lmy))))
    
    B = - np.dot(d_big.T,lmy)

    sign, logdetPostPrec = np.linalg.slogdet(postCov)
    
    C = 1/2*logdetPostPrec
    
    D = np.dot(lamb.T, (np.log(lamb) - np.ones(len(ybar))))

    return A+B+C+D

def dualProblem_grad(lamb, ybar, C_big, K_big, K_bigInv, d_big):
    postCov, postPrec = VIPostCov(K_bigInv,C_big,lamb)
    lmy = lamb-ybar
    grad = np.dot(C_big.T,np.dot(K_big,np.dot(C_big,lmy))) - d_big + np.log(lamb) - 1/2*np.diag(np.dot(C_big.T,np.dot(postCov,C_big)))
    return grad

#======================
def dualProblemRho(rho, ybar, C_big, K_big, K_bigInv, d_big):
    '''
    For optimizing dual variationa inference cost function w.r.t. log(lamb) so that
    we can disregard the optimization bounds.
    '''
    lamb = np.exp(rho)
    varMean = VIPostMean(K_big,C_big,ybar,lamb)
    
    postCov, postPrec = VIPostCov(K_bigInv,C_big,lamb)

    lmy = lamb-ybar
    
    A = 1/2*np.dot(lmy.T,np.dot(C_big.T,np.dot(K_big,np.dot(C_big,lmy))))
    
    B = - np.dot(d_big.T,lmy)

    sign, logdetPostPrec = np.linalg.slogdet(postCov)
    
    C = 1/2*logdetPostPrec
    
    D = np.dot(lamb.T, (np.log(lamb) - np.ones(len(ybar))))

    return A+B+C+D

def dualProblemRho_grad(rho, ybar, C_big, K_big, K_bigInv, d_big):
    '''
    For optimizing dual variationa inference cost function w.r.t. log(lamb) so that
    we can disregard the optimization bounds.
    '''
    lamb = np.exp(rho)
    postCov, postPrec = VIPostCov(K_bigInv,C_big,lamb)
    lmy = lamb-ybar
    grad = np.dot(C_big.T,np.dot(K_big,np.dot(C_big,lmy))) - d_big + np.log(lamb) - 1/2*np.diag(np.dot(C_big.T,np.dot(postCov,C_big)))
    grad = grad * np.exp(rho)
    return grad

#======================
def dualVariational(experiment, params, optimizeLogLambda = False, prevOptimRes = None, returnOptimRes = True, verbose = False):
    '''
    varInfRes, -post_lik, var_lowerBound = dualVariational(experiment, params)
    '''
    ridge = 0

    [ydim,T] = np.shape(experiment.data[0]['Y'])
    [ydim, xdim] = np.shape(params['C'])
    numTrials = len(experiment.data)
    trialDur = experiment.trialDur
    binSize = experiment.binSize

    C_big, d_big = util.makeCd_big(params, T)
    K_big, K = util.makeK_big(params, trialDur, binSize)
    K_bigInv = np.linalg.inv(K_big)

    x_post_mean = []
    x_post_cov = []
    x_vsmGP = []
    x_vsm = []

    post_lik = 0
    var_lowerBound = 0

    varOptimRes = []

    # =====================================================================
    # Bounded optimization over (0,inf) of lambda using the l-BFGS-b method. 
    # =====================================================================
    if optimizeLogLambda == False:
        for tr in range(numTrials):
            if verbose: print('dual variational inference trajectory of trial ' +str(tr+1) + '...')
            y = experiment.data[tr]['Y']
            ybar = np.ndarray.flatten(np.reshape(y, ydim*T))

            if prevOptimRes == None:
                lambInit = np.zeros(ydim*T)+0.5
            else:
                lambInit = prevOptimRes[tr]

            if False:
                grad = op.check_grad(dualProblemRho, dualProblemRho_grad, rhoInit, ybar, C_big, K_big, K_bigInv, d_big)
                # varinfGradCheck.append(grad)
                print('var inf grad check = '+ str(gradCheck))
                
                # Plot Gradients
                apprxGrad = op.approx_fprime(lambInit, dualProblem, 1e-8*np.ones(ydim*T), ybar, C_big, K_big, K_bigInv, d_big)
                calcdGrad = dualProblem_grad(lambInit,ybar, C_big, K_big, K_bigInv, d_big)
                plt.plot(apprxGrad,linewidth = 10, color = 'k', alpha = 0.4)
                plt.plot(calcdGrad,linewidth = 2, color = 'k', alpha = 0.4)
                plt.legend(['approximated','calculated'])
                plt.title('Approx. vs. calculated Grad of dual problem')
                plt.tight_layout()
                plt.show()
                pdb.set_trace()

            lbnd = 1e-10 # \approx 0
            VIresBFGS = op.fmin_l_bfgs_b(
                func = dualProblem,
                x0 = lambInit,
                fprime = dualProblem_grad,
                args = (ybar, C_big, K_big, K_bigInv, d_big),
                approx_grad = False,
                bounds = [(lbnd,None)]*(ydim*T),
                factr = 1e7,
                disp = False)
            optimLamb = VIresBFGS[0]
            varLowBndTr = VIresBFGS[1]
            varOptimRes.append(VIresBFGS[0])
        
            x_post_mean.append(np.reshape(VIPostMean(K_big, C_big, ybar, optimLamb), [xdim,T]))
            PostCovGP, PostPrecGP = VIPostCov(K_bigInv, C_big, optimLamb)
            x_post_cov.append(PostCovGP)
            
            post_lik = post_lik + negLogPosteriorUnNorm(VIPostMean(K_big, C_big, ybar, optimLamb), ybar, C_big, d_big, K_bigInv, xdim, ydim)
            var_lowerBound = var_lowerBound + varLowBndTr

            # Stack each xdim's covariance matrix on top of each other 
            temp_vsmGP = np.zeros([T,T,xdim])
            for kk in range(xdim):
                temp_vsmGP[:,:,kk] = PostCovGP[kk*T:(kk+1)*T, kk*T:(kk+1)*T]
            x_vsmGP.append(temp_vsmGP)

            # take main, off diagonal entries only
            temp_vsm = np.zeros([T,xdim,xdim])
            for kk in range(T):
                temp_vsm[kk][:,:] = PostCovGP[kk::T,kk::T]
            x_vsm.append(temp_vsm)

    # =====================================================================
    # Optimization w.r.t. log(lambda) so that we can optimize over \R^{qT}
    # It is ~2x slower than bounded optimization using l-bfgs-b method.
    # =====================================================================
    if optimizeLogLambda == True:
        for tr in range(numTrials):
            if verbose: print('dual variational inference trajectory of trial ' +str(tr+1) + '...')
            y = experiment.data[tr]['Y']
            ybar = np.ndarray.flatten(np.reshape(y, ydim*T))

            if prevOptimRes == None:
                rhoInit = np.zeros(ydim*T)
            else:
                rhoInit = prevOptimRes[tr]

            
            # Plot Gradients
            if False:
                grad = op.check_grad(dualProblemRho, dualProblemRho_grad, rhoInit, ybar, C_big, K_big, K_bigInv, d_big)
                varinfGradCheck.append(grad)
                print('var inf grad check = '+ str(gradCheck))

                apprxGrad = op.approx_fprime(lambInit, dualProblem, 1e-8*np.ones(ydim*T), ybar, C_big, K_big, K_bigInv, d_big)
                calcdGrad = dualProblem_grad(lambInit,ybar, C_big, K_big, K_bigInv, d_big)
                plt.plot(apprxGrad,linewidth = 10, color = 'k', alpha = 0.4)
                plt.plot(calcdGrad,linewidth = 2, color = 'k', alpha = 0.4)
                plt.legend(['approximated','calculated'])
                plt.title('Approx. vs. calculated Grad of dual problem')
                plt.tight_layout()
                plt.show()
                pdb.set_trace()

            # resVar = op.minimize(
            #     fun = dualProblemRho,
            #     x0 = rhoInit,
            #     method = 'BFGS',
            #     jac = dualProblemRho_grad,
            #     args = (ybar, C_big, K_big, K_bigInv, d_big),
            #     options = {'disp': False, 'maxiter': 200})
            # optimLamb = np.exp(resVar.x)
            # varLowBndTr = resVar.fun
            # varOptimRes.append(resVar.x)

            VIres = op.fmin_l_bfgs_b(
                func = dualProblemRho,
                x0 = rhoInit,
                fprime = dualProblemRho_grad,
                args = (ybar, C_big, K_big, K_bigInv, d_big),
                disp = False)
            optimLamb = np.exp(VIres[0])
            varLowBndTr = VIres[1]
            varOptimRes.append(VIres[0])

            x_post_mean.append(np.reshape(VIPostMean(K_big, C_big, ybar, optimLamb), [xdim,T]))
            PostCovGP, PostPrecGP = VIPostCov(K_bigInv, C_big, optimLamb)
            PostCovGP = PostCovGP + ridge*np.diag(np.ones(xdim*T))
            x_post_cov.append(PostCovGP)
            
            post_lik = post_lik + negLogPosteriorUnNorm(VIPostMean(K_big, C_big, ybar, optimLamb), ybar, C_big, d_big, K_bigInv, xdim, ydim)
            var_lowerBound = var_lowerBound + varLowBndTr

            # Stack each xdim's covariance matrix on top of each other 
            temp_vsmGP = np.zeros([T,T,xdim])
            for kk in range(xdim):
                temp_vsmGP[:,:,kk] = PostCovGP[kk*T:(kk+1)*T, kk*T:(kk+1)*T]
            x_vsmGP.append(temp_vsmGP)

            # take main, off diagonal entries only
            temp_vsm = np.zeros([T,xdim,xdim])
            for kk in range(T):
                temp_vsm[kk][:,:] = PostCovGP[kk::T,kk::T]
            x_vsm.append(temp_vsm)

    post_lik = post_lik / numTrials
    var_lowerBound = var_lowerBound / numTrials
    varInfRes = {
        'post_mean': x_post_mean,
        'post_cov' : x_post_cov,
        'post_vsm': x_vsm,
        'post_vsmGP': x_vsmGP}

    if returnOptimRes == True:
        return varInfRes, -post_lik, var_lowerBound, varOptimRes
    else:
        return varInfRes, -post_lik, var_lowerBound

