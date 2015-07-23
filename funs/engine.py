'''Module containing core functions to perform the P-GPFA fit.

.. module:: engine
    :synopsis: A useful module indeed.

.. moduleauthor: Hooram Nam <hooram.nam@tuebingen.mpg.de>

'''
import inference
import learning
import util

import numpy as np
import scipy.io as sio
import scipy.optimize as op
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

import time
import sys
import pdb

class PPGPFAfit():
    '''
    Poisson-GPFA model fit given a neural population spike data. 

    Input Attributes:
    =================
      * experiment : (util.dataset object), required
        - A dataset object with the following attributes:
            experiment.data     - A list of dictionaries in the following format:
              experiment.data[trial]['Y'] - numpy array of shape (#time bins, # neurons)
            experiment.T        - number of time bins, all trials must have the same length
            experiment.trialDur - duration of each trial in ms
            experiment.binSize  - size of bin in ms
      
      * initParams : (dict), required: initial parameter. 
        - Has the following fields:
            initParams['C']   - a numpy array of shape (#neurons, #latent dimension to fit)
            initParams['d']   - a numpy array of shape (#neurons)
            initParams['tau'] - a numpy array of shape (#latent dimension), in seconds
      
      * inferenceMethod : (str), optional
        - Specifies the posterior Gaussian approximation method used in inference. Defaults to 'laplace'.
            inferenceMethod = 'laplace' - uses laplace approximation (mean ~= mode)
            inferenceMethod = 'variational' - uses variational inference
      
      * maxEMiter : (int), optional
        - number of maximum EM iteration, defaults to 50.
      
      * EMmode : (str), optional
        - If EMmode = 'Batch', performs batch EM, where inference is performed on all available trials.
        - If EMmode = 'Online', performs online EM, where inference is performed only on smaller number of
          subsampled trials. User can specify further details of online EM via the init attributes
          onlineParamUpdateMethod and priorCovOpts.

      * onlineParamUpdateMethod : (str)
        - If 'balancingGamma', parameters are updated according to 
            params_{n+1} = (gamma[n])*params_{n} + (1-gamma[n])*argmax_{params}(M_step_cost_function(params)).
        - If 'sequentialAverage', parameters are updated according to 
            params_{n+1} = (params_{n} + argmax_{params}(M_step_cost_function(params)))/2.
        - If 'fullyUpdateAll', parameters are updated according to
            params_{n+1} = argmax_{params}(M_step_cost_function(params)).
        - If 'gradientDescent', parameters are updated according to
            params_{n+1} = params_{n} + stepSize*inv(Hessian_{params_{n}})*Gradient_{params_{n}}.
        - If 'fullyUpdateWithPrior', parameters are updated according to
            params_{n+1} = argmax_{params}(M_step_cost_function_with_prior(params, prior)).
          prior is specified by the attribute priorCovOpts.
        -- gamma is a linearly spaced decreasing sequence of length maxEMiter ranging from 0 to 1.
      
      * forceMaxIter : (bool), optional
        - If True, EM iterations continue even after convergence criteria are met. Defaults to False.
          Effective only if self.EMmode = 'Batch'.
      
      * verbose : (bool), optional 
        - If True, the fitting process is printed in the console.
        
    Resulting Attributes:
    =====================
      * optimParams - (dict), optimal parameter found
      
      * paramSeq - (list), a list containing the parameters found in each EM iteration
      
      * infRes - (dict), contains the information about inferred latent trajectories.
          infRes['post_mean'][tr] - a numpy array of shape (xdim,T). 
                                    The inferred latent trajectory of trial tr.
          infRes['post_cov'][tr] - a numpy array of shape (xdim*T,xdim*T).
                                   The covariance of the inferred latent trajectory of trial tr.
      
      * posteriorLikelihood - (list), poterior likelihood at each EM iteration.
      
      * variationalLowerBound - (list), variational lower bound at each EM iteration. 
          This attribute only exists if inferenceMethod = 'variational'.

    Resulting Methods:
    ==================
      * plotTrajectory(tr) - plots the inferred trajectory and spike counts of trial tr.
      * plotTrajectories() - plots the inferred trajectory of all trials.
      * plotParamSeq() - plots some information about how the parameters change through EM iter.
      * plotOptimParams() - plots the optimal parameters found.
      * plotFitDetails() - plots some information about the fitting process as functions of EM iter.
    '''
    # np.seterr(all='ignore')
    def __init__(self, 
        experiment, 
        initParams = None,
        xdim = 2,
        inferenceMethod = 'laplace',
        maxEMiter = 50,
        optimLogLamb = False,
        CdOptimMethod = 'TNC',
        tauOptimMethod = 'TNC',
        verbose = False,
        EMmode = 'Online',
        batchSize = 5,
        onlineParamUpdateMethod = 'diag',
        hessTol = None,
        stepPow = 0.75,
        updateCdJointly = True,
        fullyUpdateTau = False,
        extractAllTraj = False,
        extractAllTraj_trueParams = False,
        getPredictionErr = False,
        CdMaxIter = None,
        tauMaxIter = None):
        

        self.experiment = experiment

        ydim, T = np.shape(experiment.data[0]['Y'])
        trialDur = experiment.trialDur
        numTrials = len(experiment.data)
        binSize = experiment.binSize

        if initParams == None:
            initParams = util.initializeParams(xdim, ydim, experiment)
        else:
            _,xdim = np.shape(initParams['C'])

        posteriorLikelihood = []
        variationalLowerBound = []
        learningDetails = []

        params = initParams
        paramSeq = []
        paramSeq.append(initParams)

        learningTime = [] # for profiling
        inferenceTime = []


        #!BatchEM
        if EMmode == 'Batch':
            print('+-------------------- Fit Options --------------------+')
            util.Printer.stdout((str(xdim)+' |').rjust(int(55)))
            util.Printer.stdout('| Dimensionality of Latent State: ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(ydim)+' |').rjust(int(55)))
            util.Printer.stdout('| Dimensionality of Observed State (# neurons): ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(EMmode)+' |').rjust(int(55)))
            util.Printer.stdout('| EM mode: ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(maxEMiter)+' |').rjust(int(55)))
            util.Printer.stdout('| Max EM iterations: ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(inferenceMethod)+' |').rjust(int(55)))
            util.Printer.stdout('| Inference Method: ')
            sys.stdout.flush()
            print()
            print('+-----------------------------------------------------+')
            # EM loop
            for i in range(maxEMiter):
                # E step
                before = time.time()
                if inferenceMethod == 'laplace':
                    # if/else to use previous optimization result as initialization in inference 
                    if i == 0:
                        infRes, nll, lapOptimRes = inference.laplace(
                            experiment = experiment,
                            params = params, 
                            prevOptimRes = None, 
                            verbose = verbose)
                    else:
                        infRes, nll, lapOptimRes = inference.laplace(
                            experiment = experiment, 
                            params = params, 
                            prevOptimRes = lapOptimRes,
                            verbose = verbose)
                    posteriorLikelihood.append(nll)

                if inferenceMethod == 'variational':
                    # if/else to use previous optimization result as initialization for next iteration
                    if i == 0:
                        infRes, nll, vlb, varOptimRes = inference.dualVariational(
                            experiment = experiment, 
                            params = params, 
                            optimizeLogLambda = optimLogLamb, 
                            prevOptimRes = None, 
                            verbose = verbose)
                    else:
                        infRes, nll, vlb, varOptimRes = inference.dualVariational(
                            experiment = experiment, 
                            params = params, 
                            optimizeLogLambda = optimLogLamb, 
                            prevOptimRes = varOptimRes, 
                            verbose = verbose)
                    posteriorLikelihood.append(nll)
                    variationalLowerBound.append(vlb)
                after = time.time()
                inferenceTime.append(after-before)

                # M step
                before = time.time()
                params, learnDet = learning.updateParams(
                    oldParams = params, 
                    infRes = infRes, 
                    experiment = experiment, 
                    CdOptimMethod = CdOptimMethod)
                after = time.time()
                learningTime.append(after-before)
                learningDetails.append(learnDet)
                paramSeq.append(params)
                                
                # print message
                if inferenceMethod == 'laplace':
                    output = 'Iteration: %3d of %3d, nPLL: = %.4f'%(i+1,maxEMiter,nll)
                if inferenceMethod == 'variational':
                    output = 'Iteration: %3d of %3d, nPLL: = %.4f, VLB = %.4f'\
                    %(i+1,maxEMiter,nll,vlb)
                util.Printer(output)
        #!endBatchEM


        #!onlineEM
        if EMmode == 'Online':
            print('+-------------------- Fit Options --------------------+')
            util.Printer.stdout((str(xdim)+' |').rjust(int(55)))
            util.Printer.stdout('| Dimensionality of Latent State: ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(ydim)+' |').rjust(int(55)))
            util.Printer.stdout('| Dimensionality of Observed State (# neurons): ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(EMmode)+' |').rjust(int(55)))
            util.Printer.stdout('| EM mode: ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(maxEMiter)+' |').rjust(int(55)))
            util.Printer.stdout('| Max EM iterations: ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(inferenceMethod)+' |').rjust(int(55)))
            util.Printer.stdout('| Inference Method: ')
            sys.stdout.flush()
            print()
            util.Printer.stdout(('`'+str(onlineParamUpdateMethod)+'`'+' |').rjust(int(55)))
            util.Printer.stdout('| Online Param Update Method: ')
            sys.stdout.flush()
            print()
            util.Printer.stdout((str(batchSize)+' |').rjust(int(55)))
            util.Printer.stdout('| Batch size (trials): ')
            sys.stdout.flush()
            print()
            print('+-----------------------------------------------------+')

            gamma = np.linspace(0,1,maxEMiter)            
            regularizer_stepsize_Cd = 1/(np.arange(maxEMiter)+1)**(stepPow)
            regularizer_stepsize_tau = 1/(np.arange(maxEMiter)+1)**(stepPow)
            grad_descent_stepsize = 1/(np.arange(maxEMiter)+1)**stepPow
            self.invPriorCovs = []
            self.cumHess = []

            # depending on whether updating C,d jointly or not, the size of the hessian to store differs
            if updateCdJointly==True: self.invPriorCovs.append(np.diag(np.ones(xdim*ydim+ydim)))
            if updateCdJointly==False: self.invPriorCovs.append(np.diag(np.ones(xdim*ydim)))
            if updateCdJointly==True: self.cumHess.append(np.diag(np.ones(xdim*ydim+ydim)))
            if updateCdJointly==False: self.cumHess.append(np.diag(np.ones(xdim*ydim)))

            # EM Loop
            seenTrialIdx = []
            for n in range(maxEMiter):                
                # E-step
                # stochasticity in online learning comes from subsampling trials
                subsampledDat = util.subsampleTrials(experiment, batchSize)
                seenTrialIdx.append(subsampledDat.batchTrIdx)
                seenDat = util.seenTrials(experiment, seenTrialIdx)
                before = time.time()
                if inferenceMethod == 'laplace':
                    infRes, nll, lapOptimRes = inference.laplace(
                        experiment = subsampledDat, 
                        params = params, 
                        verbose = verbose)
                    posteriorLikelihood.append(nll)
                if inferenceMethod == 'variational':
                    infRes, nll, vlb, varOptimRes = inference.dualVariational(
                        experiment = subsampledDat, 
                        params = params, 
                        optimizeLogLambda = optimLogLamb, 
                        verbose = verbose)
                    posteriorLikelihood.append(nll)
                    variationalLowerBound.append(vlb)
                after = time.time()
                inferenceTime.append(after-before)
                # M-step
                before = time.time()
                # Variants of naive online learning methods
                if onlineParamUpdateMethod == 'balancingGamma':
                    newParams, learnDet = learning.updateParams(
                        oldParams = params, 
                        infRes = infRes, 
                        experiment = subsampledDat,
                        CdOptimMethod = CdOptimMethod, 
                        CdMaxIter = CdMaxIter, 
                        tauMaxIter = None,
                        verbose = verbose)
                    nextParams = newParams
                    nextParams['C'] = (gamma[n])*params['C'] + (1-gamma[n])*newParams['C']
                    nextParams['d'] = (gamma[n])*params['d'] + (1-gamma[n])*newParams['d']
                    nextParams['tau'] = (gamma[n])*params['tau'] + (1-gamma[n])*newParams['tau']
                if onlineParamUpdateMethod == 'sequentialAverage':
                    newParams, learnDet = learning.updateParams(
                        oldParams = params, 
                        infRes = infRes, 
                        experiment = subsampledDat,
                        CdOptimMethod = CdOptimMethod, 
                        CdMaxIter = CdMaxIter, 
                        tauMaxIter = None,
                        verbose = verbose)
                    nextParams = newParams
                    nextParams['C'] = (params['C'] + newParams['C'])/2
                    nextParams['d'] = (params['d'] + newParams['d'])/2                    
                    nextParams['tau'] = (params['tau'] + newParams['tau'])/2
                if onlineParamUpdateMethod == 'fullyUpdateAll':
                    newParams, learnDet = learning.updateParams(
                        oldParams = params, 
                        infRes = infRes, 
                        experiment = subsampledDat,
                        CdOptimMethod = CdOptimMethod, 
                        CdMaxIter = CdMaxIter, 
                        tauMaxIter = None,
                        verbose = verbose)
                    nextParams = newParams
                
                # online methods discussed in the paper
                if onlineParamUpdateMethod == 'hess':
                    newParams, learnDet, priorCov = learning.updateParamsWithPrior(
                        oldParams = params, 
                        infRes = infRes, 
                        experiment = subsampledDat, 
                        CdOptimMethod = CdOptimMethod, 
                        tauOptimMethod = tauOptimMethod,
                        regularizer_stepsize_Cd = regularizer_stepsize_Cd[n],
                        regularizer_stepsize_tau = regularizer_stepsize_tau[n],
                        prevInvPriorCov = self.invPriorCovs[-1],
                        covOpts = 'useHessian',
                        verbose = verbose,
                        updateCdJointly = updateCdJointly,
                        hessTol = hessTol)
                    nextParams = newParams
                    self.invPriorCovs.append(priorCov)
                if onlineParamUpdateMethod == 'diag':
                    newParams, learnDet, priorCov = learning.updateParamsWithPrior(
                        oldParams = params, 
                        infRes = infRes, 
                        experiment = subsampledDat, 
                        CdOptimMethod = CdOptimMethod, 
                        tauOptimMethod = tauOptimMethod,
                        regularizer_stepsize_Cd = regularizer_stepsize_Cd[n],
                        regularizer_stepsize_tau = regularizer_stepsize_tau[n],
                        prevInvPriorCov = self.invPriorCovs[-1],
                        covOpts = 'useDiag',
                        verbose = verbose,
                        updateCdJointly = updateCdJointly,
                        hessTol = hessTol)
                    nextParams = newParams
                    self.invPriorCovs.append(priorCov)
                if onlineParamUpdateMethod == 'grad':
                    newParams, learnDet, hess = learning.updateParamsWithGradDescent(
                        oldParams = params, 
                        infRes = infRes, 
                        experiment = subsampledDat, 
                        stepSize = grad_descent_stepsize[n],
                        cumHess = self.cumHess[-1],
                        regularizer_stepsize_tau = regularizer_stepsize_tau[n], 
                        tauOptimMethod = tauOptimMethod, 
                        verbose=verbose,
                        updateCdJointly = updateCdJointly, 
                        hessTol = hessTol)
                    self.cumHess.append(self.cumHess[-1] + hess)
                    nextParams = newParams

                # for compatibility with some test scripts
                # if onlineParamUpdateMethod == 'gradientDescent':
                #     newParams, learnDet, hess = learning.updateParamsWithGradDescent(
                #         oldParams = params, 
                #         infRes = infRes, 
                #         experiment = subsampledDat, 
                #         stepSize = grad_descent_stepsize[n],
                #         cumHess = self.cumHess[-1],
                #         regularizer_stepsize_tau = regularizer_stepsize_tau[n], 
                #         tauOptimMethod = tauOptimMethod, 
                #         verbose=verbose,
                #         updateCdJointly = updateCdJointly, 
                #         hessTol = hessTol)
                #     self.cumHess.append(self.cumHess[-1] + hess)
                #     nextParams = newParams
                # if onlineParamUpdateMethod == 'fullyUpdateWithPrior':
                #     newParams, learnDet, priorCov = learning.updateParamsWithPrior(
                #         oldParams = params, 
                #         infRes = infRes, 
                #         experiment = subsampledDat, 
                #         CdOptimMethod = CdOptimMethod, 
                #         tauOptimMethod = tauOptimMethod,
                #         regularizer_stepsize_Cd = regularizer_stepsize_Cd[n],
                #         regularizer_stepsize_tau = regularizer_stepsize_tau[n],
                #         prevInvPriorCov = self.invPriorCovs[-1],
                #         covOpts = priorCovOpts,
                #         verbose = verbose,
                #         updateCdJointly = updateCdJointly,
                #         hessTol = hessTol)
                #     nextParams = newParams
                #     self.invPriorCovs.append(priorCov)

                after = time.time()
                learningTime.append(after-before)
                if fullyUpdateTau == True:
                    nextParams['tau'] = newParams['tau']

                # print message
                if inferenceMethod == 'laplace':
                    output = 'Iteration: %3d of %3d, nPLL: = %.4f'%(n+1,maxEMiter,nll)
                if inferenceMethod == 'variational':
                    output = 'Iteration: %3d of %3d, nPLL: = %.4f, VLB = %.4f'\
                    %(n+1,maxEMiter,nll,vlb)
                util.Printer(output)

                learningDetails.append(learnDet)
                params = nextParams  # for next iteration
                paramSeq.append(params)                
            self.onlineParamUpdateMethod = onlineParamUpdateMethod
        #!endOnlineEM

        # self.experiment = experiment
        self.xdim = xdim
        self.ydim = ydim
        self.trialDur = trialDur
        self.numTrials = numTrials
        self.binSize = binSize
        self.T = T
        self.maxEMiter = maxEMiter
        self.EMmode = EMmode
        self.inferenceMethod = inferenceMethod
        self.initParams = initParams
        self.paramSeq = paramSeq
        self.posteriorLikelihood = posteriorLikelihood
        self.variationalLowerBound = variationalLowerBound
        self.learningDetails = learningDetails
        self.optimParams = params
        self.infRes = infRes # of the last batch processed in online EM
        self.processParamResults()
        self.performSpikeCountAnalysis()
        self.learningTime = np.asarray(learningTime)
        self.inferenceTime = np.asarray(inferenceTime)
        self.CdOptimMethod = CdOptimMethod
        self.optimLogLamb = optimLogLamb

        if extractAllTraj:
            self.extractTrajectories(method = inferenceMethod)
        if extractAllTraj_trueParams:
            self.extractTrajWithTrueParams(method = inferenceMethod)
        if getPredictionErr:
            self.leaveOneOutPrediction()

    def performSpikeCountAnalysis(self):
        E_y_init_params, E_yy_init_params = util.getMeanCovYfromParams(self.initParams, self.experiment)
        E_y_optim_params, E_yy_optim_params = util.getMeanCovYfromParams(self.optimParams, self.experiment)
        util.dataset.getAllRaster(self.experiment)
        E_y_obs = np.mean(self.experiment.all_raster,1)
        E_yy_obs = np.cov(self.experiment.all_raster)
        if hasattr(self.experiment, 'params'):
            E_y_true_params, E_yy_true_params = util.getMeanCovYfromParams(self.experiment.params, self.experiment)
            self.E_y_true_params = E_y_true_params
            self.E_yy_true_params = E_yy_true_params
            self.mean_err_optim_true = np.dot(E_y_true_params - E_y_optim_params,E_y_true_params - E_y_optim_params)/np.var(E_y_true_params)/self.numTrials
            self.mean_err_init_true = np.dot(E_y_true_params - E_y_init_params,E_y_true_params - E_y_init_params)/np.var(E_y_true_params)/self.numTrials
            self.cov_err_optim_true = np.linalg.norm(E_yy_true_params-E_yy_optim_params)/np.linalg.norm(E_yy_obs)
            self.cov_err_init_true = np.linalg.norm(E_yy_true_params-E_yy_init_params)/np.linalg.norm(E_yy_obs)
            self.JSdiv_cov_optim_true = util.JSLogdetDiv(E_yy_optim_params, E_yy_true_params)
            self.JSdiv_cov_init_true = util.JSLogdetDiv(E_yy_init_params, E_yy_true_params)

        self.E_y_init_params = E_y_init_params
        self.E_y_optim_params = E_y_optim_params
        self.E_yy_init_params = E_yy_init_params
        self.E_yy_optim_params = E_yy_optim_params
        self.E_y_obs = E_y_obs
        self.E_yy_obs = E_yy_obs

        self.mean_err_optim_obs = np.dot(E_y_obs - E_y_optim_params,E_y_obs - E_y_optim_params)/np.var(E_y_obs)/self.numTrials
        self.mean_err_init_obs = np.dot(E_y_obs - E_y_init_params,E_y_obs - E_y_init_params)/np.var(E_y_obs)/self.numTrials
        self.cov_err_optim_obs = np.linalg.norm(E_yy_obs-E_yy_optim_params)/np.linalg.norm(E_yy_obs)
        self.cov_err_init_obs = np.linalg.norm(E_yy_obs-E_yy_init_params)/np.linalg.norm(E_yy_obs)
        self.JSdiv_cov_optim_obs = util.JSLogdetDiv(E_yy_optim_params,E_yy_obs)
        self.JSdiv_cov_init_obs = util.JSLogdetDiv(E_yy_init_params,E_yy_obs)


    def orthonormalizeTrajectories(self):
        # self.extractTrajectories()
        U,D,V = sp.linalg.svd(self.optimParams['C'])
        x_tilde = []
        for tr in range(self.numTrials):
            x_tilde.append(np.dot(np.diag(D),V.T).dot(self.infRes['post_mean'][tr]))
        self.x_tilde = np.asarray(x_tilde)
        
    def extractTrajectories(self, method = 'laplace'):
        if method == 'laplace':
            infRes, nll, OptimRes = inference.laplace(self.experiment, self.optimParams)
            self.infRes = infRes
            self.nll_all_traj = nll
        if method == 'variational':
            infRes, nll, vlb, OptimRes = inference.dualVariational(self.experiment, self.optimParams, optimizeLogLambda = self.optimLogLamb)
            self.infRes = infRes
            self.nll_all_traj = nll
            self.vlb_all_traj = vlb

    def extractTrajWithTrueParams(self, method = 'laplace'):
        if method == 'laplace':
            infRes_trueParams, nll_trueParams, OptimRes = inference.laplace(self.experiment, self.experiment.params)
            self.infRes_trueParams = infRes_trueParams
            self.nll_trueParams_all_traj = nll_trueParams
        if method == 'variational':
            infRes_trueParams, nll_trueParams, vlb_trueParams, OptimRes = inference.dualVariational(self.experiment, self.experiment.params, optimizeLogLambda = self.optimLogLamb)
            self.infRes_trueParams = infRes_trueParams
            self.nll_trueParams_all_traj = nll_trueParams
            self.vlb_trueParams_all_traj = vlb_trueParams

    def processParamResults(self):
        '''
        Creates new attributes on the PPGPFAfit object:
        - self.tauSeq: 
            sequence of estimated Taus along the EM iteration
        - self.meanSquaredErrorOverTrueVariance_SM:
            A measure to assess the learning performance of C,d.

                            1/N \sum((E[y_est] - E[y_true])^2)
            error measure = --------------------------------
                                     Var(E[y_true])

            subspace angle


            
        '''
        self.tauSeq = np.zeros([self.xdim, self.maxEMiter])
        self.expectedSpikeCountsEst = np.zeros([self.ydim, self.maxEMiter])
        self.expectedSpikeCountsEstVar = np.zeros(self.maxEMiter)
        for i in range(self.maxEMiter):
            self.tauSeq[:,i] = self.paramSeq[i]['tau']
            self.expectedSpikeCountsEst[:,i] = self.T*np.exp(1/2*np.diag(np.dot(self.paramSeq[i]['C'],self.paramSeq[i]['C'].T))+self.paramSeq[i]['d'])
            self.expectedSpikeCountsEstVar[i] = np.var(self.expectedSpikeCountsEst[:,i])
        sampleMeanSpikeCounts = np.zeros(self.ydim)
        for tr in range(self.numTrials):
            sampleMeanSpikeCounts = sampleMeanSpikeCounts + np.sum(self.experiment.data[tr]['Y'],1)
        self.sampleMeanSpikeCounts = sampleMeanSpikeCounts/self.numTrials
        self.sampleMeanSpikeCountsVar = np.var(self.sampleMeanSpikeCounts)

        if hasattr(self.experiment,'params'):
            print('\nThis dataset is a simulated dataset.\nProcessing performance against ground truth parameters...')
            self.expectedSpikeCountsTrue = self.T*np.exp(1/2*np.diag(np.dot(self.experiment.params['C'],self.experiment.params['C'].T))+self.experiment.params['d'])
            self.expectedSpikeCountsTrueVar = np.var(self.expectedSpikeCountsTrue)
            self.varESpkCountTrue_Ratios = self.expectedSpikeCountsEstVar/self.expectedSpikeCountsTrueVar
        self.varESpkCountSampleMean_Ratios = self.expectedSpikeCountsEstVar/self.sampleMeanSpikeCountsVar
        
        # Fix as in the labnote 3/25
        self.meanSquaredErrorOverTrueVariance_SM = []
        for i in range(self.maxEMiter):
            errSM = 1/self.numTrials*np.dot((self.expectedSpikeCountsEst[:,i]-self.sampleMeanSpikeCounts),(self.expectedSpikeCountsEst[:,i]-self.sampleMeanSpikeCounts))/self.sampleMeanSpikeCountsVar
            self.meanSquaredErrorOverTrueVariance_SM.append(errSM)

        if hasattr(self.experiment,'params'):
            angles = []
            for i in range(self.maxEMiter):
                angle = util.subspaceAngle(self.experiment.params['C'],self.paramSeq[i]['C'])
                angles.append(angle)
            self.subspaceAngleC = angles

        self.CabsoluteValue = np.zeros(self.maxEMiter)
        for i in range(self.maxEMiter):
            self.CabsoluteValue[i] = self.paramSeq[i]['C'].flatten().dot(self.paramSeq[i]['C'].flatten())

    def leaveOneOutPrediction(self):
        '''
        Performs leave-one-out prediction. 
        '''
        print('Performing leave-one-out cross validation...')
        params = self.optimParams
        y_pred_mode_all = []
        pred_err_mode = 0
        for tr in range(self.numTrials):
            y_pred_mode_tr = []
            for nrn in range(self.ydim):
                # Make params without neuron# nrn
                CwoNrn = np.delete(params['C'],nrn,0)
                dwoNrn = np.delete(params['d'],nrn,0)
                paramsSplit = {'C':CwoNrn, 'd':dwoNrn, 'tau':params['tau']}

                # Make params with only neuron# nrn
                C_nrn = params['C'][nrn]
                d_nrn = params['d'][nrn]

                # Make params big
                C_big, d_big = util.makeCd_big(paramsSplit,self.T)
                K_big, K = util.makeK_big(paramsSplit, self.trialDur, self.binSize)
                K_bigInv = np.linalg.inv(K_big)

                # Make data without neuron# nrn
                y = np.delete(self.experiment.data[tr]['Y'],nrn,0)
                ybar = np.ndarray.flatten(np.reshape(y, (self.ydim-1)*self.T))

                xInit = np.ndarray.flatten(np.zeros([self.xdim*self.T,1]))
                res = op.fmin_ncg(
                    f = inference.negLogPosteriorUnNorm,
                    x0 = xInit,
                    fprime = inference.negLogPosteriorUnNorm_grad,
                    fhess = inference.negLogPosteriorUnNorm_hess,
                    args = (ybar, C_big, d_big, K_bigInv, self.xdim, self.ydim-1),
                    disp = False,
                    full_output = True)

                x_post_mode = np.reshape(res[0],[self.xdim,self.T])
                y_pred_mode_nrn = np.exp(C_nrn.dot(x_post_mode).T + d_nrn)
                pred_err_mode = pred_err_mode + np.dot(self.experiment.data[tr]['Y'][nrn]-y_pred_mode_nrn,self.experiment.data[tr]['Y'][nrn]-y_pred_mode_nrn)
                y_pred_mode_tr.append(y_pred_mode_nrn)
            y_pred_mode_all.append(y_pred_mode_tr)
        self.y_pred_mode = np.asarray(y_pred_mode_all)
        self.pred_err_mode = pred_err_mode

    # Below only plots

    def plotCovAnalysis(self):
        E_y_init_params = self.E_y_init_params
        E_yy_init_params = self.E_yy_init_params
        E_y_optim_params  = self.E_y_optim_params
        E_yy_optim_params = self.E_yy_optim_params
        E_y_obs = self.E_y_obs
        E_yy_obs = self.E_yy_obs

        if hasattr(self.experiment, 'params'):
            E_y_true_params  = self.E_y_true_params
            E_yy_true_params = self.E_yy_true_params

            fig, ax = plt.subplots(ncols = 4, figsize=(12,3))
            im0=ax[0].matshow(E_yy_obs,
                          vmin = np.min([E_yy_obs,E_yy_true_params,E_yy_optim_params,E_yy_init_params]),
                          vmax = np.max([E_yy_obs,E_yy_true_params,E_yy_optim_params,E_yy_init_params]))
            im1=ax[1].matshow(E_yy_true_params,
                          vmin = np.min([E_yy_obs,E_yy_true_params,E_yy_optim_params,E_yy_init_params]),
                          vmax = np.max([E_yy_obs,E_yy_true_params,E_yy_optim_params,E_yy_init_params]))
            im2=ax[2].matshow(E_yy_init_params,
                          vmin = np.min([E_yy_obs,E_yy_true_params,E_yy_optim_params,E_yy_init_params]),
                          vmax = np.max([E_yy_obs,E_yy_true_params,E_yy_optim_params,E_yy_init_params]))
            im3=ax[3].matshow(E_yy_optim_params,
                          vmin = np.min([E_yy_obs,E_yy_true_params,E_yy_optim_params,E_yy_init_params]),
                          vmax = np.max([E_yy_obs,E_yy_true_params,E_yy_optim_params,E_yy_init_params]))
            divider0 = make_axes_locatable(ax[0])
            cax0 = divider0.append_axes('right',size='10%',pad=0.05)
            cbar0 = plt.colorbar(im0,cax=cax0)
            divider1 = make_axes_locatable(ax[1])
            cax1 = divider1.append_axes('right',size='10%',pad=0.05)
            cbar1 = plt.colorbar(im1,cax=cax1)
            divider2 = make_axes_locatable(ax[2])
            cax2 = divider2.append_axes('right',size='10%',pad=0.05)
            cbar2 = plt.colorbar(im2,cax=cax2)
            divider3 = make_axes_locatable(ax[3])
            cax3 = divider3.append_axes('right',size='10%',pad=0.05)
            cbar3 = plt.colorbar(im3,cax=cax3)
            util.simpleaxis(ax[0])
            util.simpleaxis(ax[1])
            util.simpleaxis(ax[2])
            util.simpleaxis(ax[3])
            ax[0].set_title(r'$\mathbf{Cov}[y_{obs}]$')
            ax[1].set_title(r'$\mathbf{Cov}[y|\theta_{true}]$')
            ax[2].set_title(r'$\mathbf{Cov}[y|\theta_{init}]$')
            ax[3].set_title(r'$\mathbf{Cov}[y|\theta_{optim}]$')
            plt.tight_layout()
            # plt.savefig('observed_expected_covs.pdf')
            # plt.savefig('observed_expected_covs.png')

            fig, ax = plt.subplots(ncols = 2, figsize=(10,5))
            err_optim = np.linalg.norm(E_yy_true_params-E_yy_optim_params)/np.linalg.norm(E_yy_obs)
            err_init = np.linalg.norm(E_yy_true_params-E_yy_init_params)/np.linalg.norm(E_yy_obs)
            im0=ax[0].imshow(E_yy_true_params - E_yy_optim_params,# vmin = -1, vmax = 3,
                         vmin = np.min([E_yy_true_params-E_yy_optim_params,E_yy_true_params-E_yy_init_params]),
                         vmax = np.max([E_yy_true_params-E_yy_optim_params,E_yy_true_params-E_yy_init_params]),
                         interpolation = 'nearest')
            im1=ax[1].imshow(E_yy_true_params - E_yy_init_params,#  vmin = -1, vmax = 3,
                         vmin = np.min([E_yy_true_params-E_yy_optim_params,E_yy_true_params-E_yy_init_params]),
                         vmax = np.max([E_yy_true_params-E_yy_optim_params,E_yy_true_params-E_yy_init_params]),
                         interpolation = 'nearest')
            divider0 = make_axes_locatable(ax[0])
            cax0 = divider0.append_axes('right',size='10%',pad=0.05)
            cbar0 = plt.colorbar(im0,cax=cax0)
            divider1 = make_axes_locatable(ax[1])
            cax1 = divider1.append_axes('right',size='10%',pad=0.05)
            cbar1 = plt.colorbar(im1,cax=cax1)
            ax[0].set_title(r'$\mathbf{Cov}[y|\theta_{true}] - \mathbf{Cov}[y|\theta_{optim}]$, $|\cdot|/|\mathbf{Cov}[y_{obs}]|$ = %.2f'%err_optim)
            ax[1].set_title(r'$\mathbf{Cov}[y|\theta_{true}] - \mathbf{Cov}[y|\theta_{init}]$, $|\cdot|/|\mathbf{Cov}[y_{obs}]|$ = %.2f'%err_init)
            plt.tight_layout()

        fig, ax = plt.subplots(ncols = 2, figsize=(10,5))
        err_optim = np.linalg.norm(E_yy_obs-E_yy_optim_params)/np.linalg.norm(E_yy_obs)
        err_init = np.linalg.norm(E_yy_obs-E_yy_init_params)/np.linalg.norm(E_yy_obs)
        im0=ax[0].imshow(E_yy_obs - E_yy_optim_params,# vmin = -1, vmax = 3,
                     vmin = np.min([E_yy_obs-E_yy_optim_params,E_yy_obs-E_yy_init_params]),
                     vmax = np.max([E_yy_obs-E_yy_optim_params,E_yy_obs-E_yy_init_params]),
                     interpolation = 'nearest')
        im1=ax[1].imshow(E_yy_obs - E_yy_init_params,#  vmin = -1, vmax = 3,
                     vmin = np.min([E_yy_obs-E_yy_optim_params,E_yy_obs-E_yy_init_params]),
                     vmax = np.max([E_yy_obs-E_yy_optim_params,E_yy_obs-E_yy_init_params]),
                     interpolation = 'nearest')
        divider0 = make_axes_locatable(ax[0])
        cax0 = divider0.append_axes('right',size='10%',pad=0.05)
        cbar0 = plt.colorbar(im0,cax=cax0)
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes('right',size='10%',pad=0.05)
        cbar1 = plt.colorbar(im1,cax=cax1)
        ax[0].set_title(r'$\mathbf{Cov}[y_{obs}] - \mathbf{Cov}[y|\theta_{optim}]$, $|\cdot|/|\mathbf{Cov}[y_{obs}]|$ = %.2f'%err_optim)
        ax[1].set_title(r'$\mathbf{Cov}[y_{obs}] - \mathbf{Cov}[y|\theta_{init}]$, $|\cdot|/|\mathbf{Cov}[y_{obs}]|$ = %.2f'%err_init)
        plt.tight_layout()

    def plotLNOprediction(self,trialToPlot = 0, neuronToPlot = 0):
        plt.plot(self.y_pred_mode[trialToPlot,neuronToPlot],linewidth = 3)
        plt.stem(self.experiment.data[trialToPlot]['Y'][neuronToPlot])
        plt.title('Leave One Out Prediction: trial '+ str(trialToPlot+1)+' neuron ' + str(neuronToPlot+1))
        plt.xlabel('Time ('+str(self.binSize)+' ms bin)')
        plt.ylabel('Spike Counts')
        plt.tight_layout()

    def plotTrajectory(self, trial = 0):
        '''
        Makes a figure with three subplots: 
            1. Extracted latent trajectory
            2. Ground truth latent trajectory
            3. Spike counts
        '''
        xdim = self.xdim
        ydim = self.ydim
        T = self.T

        if hasattr(self.experiment,'xdim'):
            fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (4,6), sharex=True)
            traj = self.infRes['post_mean'][trial].T
            for xd in range(xdim):
                traj_temp = traj[:,xd]
                err = np.diag(self.infRes['post_vsmGP'][trial][:,:,xd])**(0.5)
                ax0.plot(range(T), traj_temp)
                ax0.fill_between(range(T), traj_temp-err, traj_temp+err, alpha = 0.1, edgecolor = '#64b5f6', facecolor = '#64b5f6')
            if self.inferenceMethod == 'variational':
                ax0.set_title('Latent trajectory inferred with variational')
            if self.inferenceMethod == 'laplace':
                ax0.set_title('Latent trajectory inferred with Laplace')
            ax1.plot(self.experiment.data[trial]['X'].T)
            ax1.set_title('Ground truth trajectroy')
            raster = ax2.imshow(self.experiment.data[trial]['Y'], interpolation="nearest", aspect = 'auto')
            raster.set_cmap('Greys')
            ax2.set_title('Spike counts')
            ax2.set_ylabel('Neuron Index')
            ax2.set_xlabel('Time (' + str(self.binSize) + ' ms bin)')
            ax0.grid(which='both')
            ax1.grid(which='both')
        else:
            fig, (ax0,ax2) = plt.subplots(nrows = 2, figsize = (4,4), sharex=True)
            traj = self.infRes['post_mean'][trial].T
            for xd in range(xdim):
                traj_temp = traj[:,xd]
                err = np.diag(self.infRes['post_vsmGP'][trial][:,:,xd])**(0.5)
                ax0.plot(range(T), traj_temp)
                ax0.fill_between(range(T), traj_temp-err, traj_temp+err, alpha = 0.1, edgecolor = '#64b5f6', facecolor = '#64b5f6')
            if self.inferenceMethod == 'variational':
                ax0.set_title('Latent trajectory inferred with variational')
            if self.inferenceMethod == 'laplace':
                ax0.set_title('Latent trajectory inferred with Laplace')
            raster = ax2.imshow(self.experiment.data[trial]['Y'], interpolation="nearest", aspect = 'auto')
            raster.set_cmap('Greys')
            ax2.set_title('Spike counts')
            ax2.set_ylabel('Neuron Index')
            ax2.set_xlabel('Time (' + str(self.binSize) + ' ms bin)')
            ax0.grid(which='both')
        plt.tight_layout()
        return fig

    def plotFitDetails(self):
        '''
        Makes a figure with the following subplots:
            1. Posterior log likelihood over the EM iteration
            2. (only for variational inference) Variational lower bound over the EM iterations
            3. Cost function for learning C and d
            4. Cost function for learning each Tau
        '''
        obsCost = np.zeros(self.maxEMiter)
        dynCost = np.zeros([self.xdim,self.maxEMiter])

        for itr in range(self.maxEMiter):
            obsCost[itr] = self.learningDetails[itr]['Cd']
            for xd in range(self.xdim):
                dynCost[xd,itr] = self.learningDetails[itr]['tau'][xd].fun

        if self.inferenceMethod == 'laplace':
            plt.figure(figsize = (5,8))
            gs = gridspec.GridSpec(3, self.xdim)

            ax_pll = plt.subplot(gs[0,:])
            ax_cdl = plt.subplot(gs[1,:],sharex = ax_pll)
            ax_taul = []
            for xd in range(self.xdim):
                ax_taul.append(plt.subplot(gs[2,xd]))
            
            ax_pll.plot(self.posteriorLikelihood,linewidth = 3,alpha = 0.7)
            ax_pll.set_title('Posterior log-likelihood')
            ax_pll.set_xlabel('EM iteration')
            ax_cdl.plot(obsCost,linewidth = 3,alpha = 0.7)
            ax_cdl.set_title('C,d learning cost')
            ax_cdl.set_xlabel('EM iteration')
            ax_pll.grid(which = 'both')
            ax_cdl.grid(which = 'both')
            for xd in range(self.xdim):
                ax_taul[xd].plot(dynCost[xd,:],linewidth = 3,alpha = 0.7)
                ax_taul[xd].set_title('Tau xdim '+str(xd+1) +'cost')
                ax_taul[xd].set_xlabel('EM iteration')
                ax_taul[xd].grid(which='both')
            # plt.tight_layout()


        if self.inferenceMethod == 'variational':
            gs = gridspec.GridSpec(4, self.xdim)

            ax_pll = plt.subplot(gs[0,:])
            ax_vlb = plt.subplot(gs[1,:])
            ax_cdl = plt.subplot(gs[2,:])
            ax_taul = []
            for xd in range(self.xdim):
                ax_taul.append(plt.subplot(gs[3,xd]))
            
            ax_pll.plot(self.posteriorLikelihood,linewidth = 3,alpha = 0.7)
            ax_pll.set_title('Posterior log-likelihood')
            ax_pll.set_xlabel('EM iteration')
            
            ax_vlb.plot(self.variationalLowerBound,linewidth = 3,alpha = 0.7)
            ax_vlb.set_title('Variational Lower Bound')
            ax_vlb.set_xlabel('EM iteration')

            ax_cdl.plot(obsCost)
            ax_cdl.set_title('C,d learning cost')
            ax_cdl.set_xlabel('EM iteration')

            ax_pll.grid(which = 'both')
            ax_cdl.grid(which = 'both')
            ax_vlb.grid(which = 'both')
            for xd in range(self.xdim):
                ax_taul[xd].plot(dynCost[xd,:],linewidth = 3,alpha = 0.7)
                ax_taul[xd].set_title('Tau xdim '+str(xd+1) +' cost')
                ax_taul[xd].set_xlabel('EM iteration')
            # plt.tight_layout()
        plt.tight_layout()

    def plotParamSeq(self):
        '''
        Plots a 2x2 grid of four plots showing the progress of the fit reflected by
            [0][0] error in spike count
            [0][1] error in subspace
            [1][0] expected & true spike counts
            [1][1] true & estimated tau
        '''
        if hasattr(self.experiment,'params'): #check if data is a simulation
            fig1,ax = plt.subplots(ncols = 2, nrows = 2, figsize = (9,6))
            ax[0][0].plot(self.meanSquaredErrorOverTrueVariance_SM,linewidth = 3,alpha = 0.7,color ='g')
            ax[0][0].set_ylabel('MSE(true-est)/Var(true)')
            ax[0][0].set_xlabel('EM iteration')
            ax[0][0].set_title('Error in Spike Count')
            ax[0][0].set_yscale('log')
            ax[0][0].grid(which='both')

            ax[0][1].plot(self.subspaceAngleC,linewidth = 3,alpha = 0.7,color ='g')
            ax[0][1].set_ylabel('Angle')
            ax[0][1].set_xlabel('EM iteration')
            ax[0][1].set_title('Error in Subspace')
            ax[0][1].grid(which='both')

            ax[1][0].plot(self.expectedSpikeCountsEst[:,-1],linewidth = 3,alpha = 0.5)
            ax[1][0].plot(self.sampleMeanSpikeCounts,linewidth=3.0, alpha = 0.5)
            ax[1][0].set_title('Estimated & True Spike Counts')
            ax[1][0].set_ylabel('Spike counts')
            ax[1][0].set_xlabel('Neuron index')
            ax[1][0].legend(['$E[y|C_{est},d_{est}]$','Mean spike count'],fontsize = 10,loc='best')
            ax[1][0].grid(which='both')

            ax[1][1].plot(range(self.maxEMiter),self.tauSeq.T, linewidth = 3, alpha = 0.9)
            ax[1][1].plot(np.ones([self.maxEMiter,self.experiment.xdim])*self.experiment.params['tau'],'k--')
            ax[1][1].set_xlabel('EM Iteration')
            ax[1][1].set_title('Tau history')
            ax[1][1].set_ylim([0,None])
            ax[1][1].set_ylabel('Time [sec]')
            ax[1][1].grid(which='both')
            plt.tight_layout()
        else: #if real data
            fig1,(ax0,ax1,ax2) = plt.subplots(ncols = 3, figsize = (12,3))
            ax0.plot(self.meanSquaredErrorOverTrueVariance_SM,linewidth = 3,alpha = 0.7,color ='g')
            ax0.set_ylabel('MSE(true-est)/Var(true)')
            ax0.set_xlabel('EM iteration')
            ax0.grid(which='both')
            ax0.set_yscale('log')
            ax0.set_title('Error in Spike Count')

            ax1.plot(self.expectedSpikeCountsEst[:,-1],linewidth = 3,alpha = 0.5)
            ax1.plot(self.sampleMeanSpikeCounts,linewidth=3.0, alpha = 0.5)
            ax1.set_title('Estimated & True Spike Counts')
            ax1.set_ylabel('Spike counts')
            ax1.set_xlabel('Neuron index')
            ax1.grid(which='both')
            ax1.legend(['$E[y|C_{est},d_{est}]$','Mean spike count'],fontsize = 10,loc='best')

            ax2.plot(range(self.maxEMiter),self.tauSeq.T, linewidth = 3, alpha = 0.9)
            # Check if dataset is simulated and if true xdim == fit xdim.
            if hasattr(self.experiment,'params'): 
                if hasattr(self.experiment,'xdim'):
                    if self.experiment.xdim == self.xdim:
                        ax2.plot(np.ones([self.maxEMiter,self.xdim])*self.experiment.params['tau'],color = 'k', linewidth = 2,alpha = 0.3)
            ax2.set_xlabel('EM Iteration')
            ax2.set_title('Tau history')
            ax2.set_ylim([0,None])
            ax2.grid(which='both')
            ax2.set_ylabel('Time [sec]')
            plt.tight_layout()


    def plotParamComparison(self):
        fig2, (ax_C, ax_d, ax_tau) = plt.subplots(nrows = 3,figsize = (4.5,8))
        ax_C.plot(self.optimParams['C'],linewidth = 3, alpha = 0.6)
        if hasattr(self.experiment,'params'): 
            ax_C.plot(self.experiment.params['C'],linewidth = 1,alpha = 0.2)
        ax_C.set_title('Estimated C')
        ax_C.set_xlabel('Neuron Index')
        
        ax_d.plot(self.optimParams['d'],linewidth = 3, alpha = 0.6)
        if hasattr(self.experiment,'params'):
            ax_d.plot(self.experiment.params['d'],linewidth = 2, alpha = 0.3)
        ax_d.set_title('Estimated d')
        ax_d.set_xlabel('Neuron Index')

        ax_tau.bar(np.arange(1,self.xdim+1)-0.25,self.optimParams['tau'],width = 0.5)
        ax_tau.set_title('Estimated Taus')
        ax_tau.set_xlabel('Latent Dimension')
        ax_tau.set_ylabel('Timescale (seconds)')
        ax_tau.set_xlim([0,self.xdim+1])
        ax_tau.set_xticks(range(1,self.xdim+1))
        plt.tight_layout()


    def plotTrajectories(self):
        if not(hasattr(self,'x_tilde')):
            if self.xdim == 1:
                fig,ax=plt.subplots(figsize=(5,5))
                for i in range(len(self.experiment.data)):    
                    ax.plot(self.infRes['post_mean'][i].T,'k')
                ax.set_title('xdim 1')
                ax.set_xlabel('Time')
                # util.simpleaxis(ax)
                plt.tight_layout()        

            if self.xdim >= 2:
                fig, ax = plt.subplots(ncols = self.xdim, sharey = True,figsize=(5,5))
                for i in range(len(self.experiment.data)):
                    for xd in range(self.xdim):
                        if hasattr(self,'x_tilde'):
                            ax[xd].plot(self.x_tilde[i][xd],'k')
                        else:
                            ax[xd].plot(self.infRes['post_mean'][i][xd],'k')
                        ax[xd].set_title('xdim ' + str(xd))
                        ax[xd].set_xlabel('Time')
                        # util.simpleaxis(ax[xd])
                plt.tight_layout()        

                if self.xdim >=3:
                    fig = plt.figure(figsize=(5,5))
                    ax = fig.gca(projection='3d')
                    for i in range(len(self.experiment.data)):
                        traj = self.infRes['post_mean'][i]
                        ax.plot(traj[0],traj[1],traj[2],'k.-')
                    ax.set_xlabel('xdim1')
                    ax.set_ylabel('xdim2')
                    ax.set_zlabel('xdim3')
                    # util.simpleaxis(ax)
        else:
            if self.xdim == 1:
                fig,ax=plt.subplots(figsize=(5,5))
                for i in range(len(self.experiment.data)):    
                    ax.plot(self.x_tilde[i].T,'k')
                ax.set_title('xdim 1')
                ax.set_xlabel('Time')
                # util.simpleaxis(ax)
                plt.tight_layout()        

            if self.xdim >= 2:
                fig, ax = plt.subplots(ncols = self.xdim, sharey = True,figsize=(5,5))
                for i in range(len(self.experiment.data)):
                    for xd in range(self.xdim):
                        if hasattr(self,'x_tilde'):
                            ax[xd].plot(self.x_tilde[i][xd],'k')
                        else:
                            ax[xd].plot(self.x_tilde[i][xd],'k')
                        ax[xd].set_title('xdim ' + str(xd))
                        ax[xd].set_xlabel('Time')
                        # util.simpleaxis(ax[xd])
                plt.tight_layout()        

                if self.xdim >=3:
                    fig = plt.figure(figsize=(5,5))
                    ax = fig.gca(projection='3d')
                    for i in range(len(self.experiment.data)):
                        traj = self.x_tilde[i]
                        ax.plot(traj[0],traj[1],traj[2],'k.-')
                    ax.set_xlabel('xdim1')
                    ax.set_ylabel('xdim2')
                    ax.set_zlabel('xdim3')

    def plotOptimParams(self):
        plt.figure(figsize=(6,4))
        gs = gridspec.GridSpec(2,2)
        ax_C = plt.subplot(gs[0,0])
        ax_d = plt.subplot(gs[1,0])
        ax_K = plt.subplot(gs[:,1])
        ax_C.plot(self.optimParams['C'], linewidth=2)
        ax_C.grid(which='both')
        ax_C.set_title('$C_{est}$')
        ax_C.set_ylabel('Latent Dimension Index')
        ax_C.set_xlabel('Neuron Index')
        ax_C.set_yticks(range(self.xdim))
        ax_d.plot(self.optimParams['d'].T)
        ax_d.grid(which='both')
        ax_d.set_title('$d_{est}$')
        ax_d.set_xlabel('Neuron Index')
        K_big,K = util.makeK_big(params = self.optimParams, trialDur = self.trialDur, binSize = self.binSize)
        im0 = ax_K.imshow(K_big, interpolation = "nearest")
        ax_K.set_title('$K(Tau_{est})$')
        divider0 = make_axes_locatable(ax_K)
        cax0 = divider0.append_axes('right',size='10%',pad=0.05)
        cbar0 = plt.colorbar(im0,cax=cax0)
        plt.tight_layout()



