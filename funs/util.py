#util.py
# A module that contains various small functions that the main engine makes use of.
import funs.learning as learning
import funs.inference as inference
import funs.engine as engine
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.optimize as op
from scipy.optimize import approx_fprime
import statsmodels.tools.numdiff as nd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import pdb
import copy
import pickle
import sys
import pandas

def JSLogdetDiv(X,Y):
    return np.log(np.linalg.det((X+Y)/2)) - 1/2*np.log(np.linalg.det(X.dot(Y)))

def getMeanCovYfromParams(params, experiment):
    T = experiment.T
    rho = params['d']
    n = len(rho)
    lamb = np.dot(params['C'],params['C'].T)
    E_y = np.exp(1/2*np.diag(lamb)+rho)
    # E_yy = np.diag(E_y) + np.diag(np.exp(np.diag(lamb))*E_y**2)
    E_yy = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i == j:
                E_yy[i,j] = E_y[i] + np.exp(lamb[i,i]/2)*E_y[i]**2
            else:
                E_yy[i,j] = E_y[i]*E_y[j]*np.exp(lamb[i,j]/2)
    # E_yy = np.dot(E_y,E_y.T)*np.exp(lamb) + np.diag(E_y)
    return E_y, E_yy

def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "-"

def raster(event_times_list, color='k'):
    """
    Creates a raster plot

    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines

    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax

class load_crcns_data():
    def __init__(
        self,
        filepath,
        trialDur = 1000,
        binSize = 20,
        numTrials = None):
        T = int(np.floor(trialDur/binSize))
        spikeTimes = pandas.read_pickle(filepath)
        uniqueUnits = np.unique(spikeTimes.unit.values)
        ydim = len(uniqueUnits)
        totalNumBins = int(np.floor(max(spikeTimes.time.values)/(binSize/1000)))
        numBinPerTrial = int(np.floor(trialDur/binSize))
        if numTrials == None: numTrials = int(np.floor(totalNumBins/T))
        spikeCountsAll = np.zeros([ydim, totalNumBins])
        spikeCountsTrial = np.zeros([numTrials,ydim,numBinPerTrial])
        value_time = spikeTimes.time.values
        value_unit = spikeTimes.unit.values

        times = []
        data = []
        for yd in range(ydim):
            times.append(spikeTimes.time[spikeTimes.unit == uniqueUnits[yd]])
            spikeCountsAll[yd] = np.histogram(times[-1].values,totalNumBins)[0]

        # pdb.set_trace()
        for tr in range(numTrials):
            spikeCountsTrial[tr,:,:] = spikeCountsAll[:,tr*numBinPerTrial:(tr+1)*numBinPerTrial]
            data.append({'Y':spikeCountsTrial[tr,:,:]})

        self.spikeTimes = spikeTimes
        self.numTrials = numTrials
        self.data = data
        self.ydim = ydim
        self.trialDur = trialDur
        self.binSize = binSize
        self.T = T


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_yaxis().set_tick_params(which='major', direction='out')
    ax.get_xaxis().set_tick_params(which='major', direction='out')

class Printer():
    '''print things to stdout on one line dynamically.'''
    def __init__(self,data):
        sys.stdout.write('\r\x1b[K'+data.__str__())
        sys.stdout.flush()
    def stdout(message):
        sys.stdout.write(message)
        sys.stdout.write('\b' * len(message))   # \b: non-deleting backspace
        
class loadDataForGPFA_CV_comparison():
    def __init__(self):
        matdata = sio.loadmat('data/dat.mat')
        ydim, trialDur = np.shape(matdata['dat']['spikes'][0][0][:,:-1])
        numTrials = len(matdata['dat']['spikes'][0])
        binSize = 20
        T = int(trialDur/binSize)
        data = []
        allRaster = np.zeros([ydim, T*numTrials])
        for tr in range(numTrials):
            raster = matdata['dat']['spikes'][0][tr]
            spkCts = np.zeros([ydim,T])
            for t in range(T):
                spkCts[:,t] = np.sum(raster[:,t*binSize:(t+1)*(binSize)],1)
            data.append({'Y':spkCts})
            allRaster[:,tr*T:(tr+1)*T] = spkCts
        self.ydim = ydim
        self.trialDur = trialDur
        self.binSize = binSize
        self.T = T
        self.data = data
        self.numTrials = numTrials
        self.raster = allRaster
        self.avgFR = np.sum(self.raster,1)/self.numTrials/self.trialDur*1000

class loadDataHighData():
    def __init__(self,filename = 'data/ex1_spikecounts.mat'):
        matdata = sio.loadmat(filename)
        ydim, trialDur = np.shape(matdata['D']['data'][0][0])
        binSize = 10
        T = int(trialDur/binSize)
        data = []
        numTrials = len(matdata['D']['data'][0])
        allRaster = np.zeros([ydim, T*numTrials])
        for tr in range(numTrials):
            raster = matdata['D']['data'][0][tr]
            spkCts = np.zeros([ydim,T])
            for t in range(T):
                spkCts[:,t] = np.sum(raster[:,t*binSize:(t+1)*(binSize)],1)
            data.append({'Y':spkCts})
            allRaster[:,tr*T:(tr+1)*T] = spkCts
        self.ydim = ydim
        self.trialDur = trialDur
        self.binSize = binSize
        self.T = T
        self.data = data
        self.numTrials = numTrials
        self.raster = allRaster
        self.avgFR = np.sum(self.raster,1)/self.numTrials/self.trialDur*1000

class crossValidation():
    def __init__(
        self,
        experiment,
        numTrainingTrials = 10,
        numTestTrials = 2,
        maxXdim = 6,
        maxEMiter = 3,
        batchSize = 5,
        inferenceMethod = 'laplace',
        learningMethod = 'batch'):
        print('Assessing optimal latent dimensionality will take a long time.')

        trainingSet, testSet = splitTrainingTestDataset(experiment, numTrainingTrials, numTestTrials)
        errs = []
        fits = []
        for xdimFit in np.arange(1,maxXdim+1):
            initParams = initializeParams(xdimFit, trainingSet.ydim, trainingSet)
            
            if learningMethod == 'batch':
                fit = engine.PPGPFAfit(
                    experiment = trainingSet, 
                    initParams = initParams, 
                    inferenceMethod = inferenceMethod,
                    EMmode = 'Batch', 
                    maxEMiter = maxEMiter)
                predBatch,predErr = leaveOneOutPrediction(fit.optimParams, testSet)
                errs.append(predErr)

            if learningMethod == 'diag':
                fit = engine.PPGPFAfit(
                    experiment = trainingSet, 
                    initParams = initParams, 
                    inferenceMethod = inferenceMethod,
                    EMmode = 'Online', 
                    onlineParamUpdateMethod = 'diag',
                    maxEMiter = maxEMiter,
                    batchSize = batchSize)
                predDiag,predErr = leaveOneOutPrediction(fit.optimParams, testSet)
                errs.append(predErr)
            
            if learningMethod == 'hess':
                fit = engine.PPGPFAfit(
                    experiment = trainingSet, 
                    initParams = initParams, 
                    inferenceMethod = inferenceMethod,
                    EMmode = 'Online', 
                    onlineParamUpdateMethod = 'hess',
                    maxEMiter = maxEMiter,
                    batchSize = batchSize)
                predHess,predErr = leaveOneOutPrediction(fit.optimParams, testSet)
                errs.append(predErr)
            
            if learningMethod == 'grad':
                fit = engine.PPGPFAfit(
                    experiment = trainingSet, 
                    initParams = initParams, 
                    inferenceMethod = inferenceMethod,
                    EMmode = 'Online', 
                    onlineParamUpdateMethod = 'grad',
                    maxEMiter = maxEMiter,
                    batchSize = batchSize)
                predGrad,predErr = leaveOneOutPrediction(fit.optimParams, testSet)
                errs.append(predErr)
            fits.append(fit)

        self.inferenceMethod = inferenceMethod
        self.learningMethod = learningMethod
        self.optimXdim=np.argmin(errs)+1 # because python indexes from 0 
        self.errs = errs
        self.maxXdim = maxXdim
        self.fits = fits

    def plotPredictionError(self):
        plt.figure(figsize=(5,4))
        plt.plot(np.arange(1,self.maxXdim+1),self.errs,'b.-',markersize=5,linewidth=2)
        plt.legend([self.method],fontsize=9,framealpha=0.2)
        plt.xlabel('Latent Dimensionality')
        plt.ylabel('Error')
        plt.title('Latent Dimension vs. Prediction Error')
        plt.grid(which='both')
        plt.tight_layout()

def splitTrainingTestDataset(experiment, numTrainingTrials, numTestTrials):
    if numTestTrials + numTrainingTrials > experiment.numTrials:
        print('Error: Number of training trials and test trials must sum to less than the number of available trials.')
    trainingSet = copy.copy(experiment)
    testSet = copy.copy(experiment)

    trainingSet.data = experiment.data[:numTrainingTrials]
    trainingSet.numTrials = numTrainingTrials
    
    testSet.data = experiment.data[numTrainingTrials:numTrainingTrials+numTestTrials]
    testSet.numTrials = numTestTrials
    
    return trainingSet, testSet

def plotLeaveOneOutPrediction(pred_mode, testSet, trial, neuron):
    plt.figure(figsize = (5,4))
    plt.plot(pred_mode[trial][neuron],linewidth=2)
    plt.plot(testSet.data[trial]['Y'][neuron],'.', markersize = 10)
    plt.ylim([0,max(2,pred_mode[trial][neuron].all())])
    plt.xlabel('Time ('+str(testSet.binSize)+' ms bins)')
    plt.ylabel('Spike Counts')
    plt.legend(['Prediction','True'])
    plt.title('LNO prediction, trial ' +str(trial)+', neuron '+str(neuron))
    plt.grid(which='both')
    plt.tight_layout()

def leaveOneOutPrediction(params, experiment):
    '''
    Performs leave-one-out prediction. 
    '''
    ydim, xdim = np.shape(params['C'])
    print('Performing leave-one-out cross validation...')
    y_pred_mode_all = []
    pred_err_mode = 0
    for tr in range(experiment.numTrials):
        y_pred_mode_tr = []
        for nrn in range(experiment.ydim):
            # Make params without neuron# nrn
            CwoNrn = np.delete(params['C'],nrn,0)
            dwoNrn = np.delete(params['d'],nrn,0)
            paramsSplit = {'C':CwoNrn, 'd':dwoNrn, 'tau':params['tau']}

            # Make params with only neuron# nrn
            C_nrn = params['C'][nrn]
            d_nrn = params['d'][nrn]

            # Make params big
            C_big, d_big = makeCd_big(paramsSplit,experiment.T)
            K_big, K = makeK_big(paramsSplit, experiment.trialDur, experiment.binSize)
            K_bigInv = np.linalg.inv(K_big)

            # Make data without neuron# nrn
            y = np.delete(experiment.data[tr]['Y'],nrn,0)
            ybar = np.ndarray.flatten(np.reshape(y, (experiment.ydim-1)*experiment.T))

            xInit = np.ndarray.flatten(np.zeros([xdim*experiment.T,1]))
            res = op.fmin_ncg(
                f = inference.negLogPosteriorUnNorm,
                x0 = xInit,
                fprime = inference.negLogPosteriorUnNorm_grad,
                fhess = inference.negLogPosteriorUnNorm_hess,
                args = (ybar, C_big, d_big, K_bigInv, xdim, experiment.ydim-1),
                disp = False,
                full_output = True)

            x_post_mode = np.reshape(res[0],[xdim,experiment.T])
            y_pred_mode_nrn = np.exp(C_nrn.dot(x_post_mode).T + d_nrn)
            pred_err_mode = pred_err_mode + np.dot(experiment.data[tr]['Y'][nrn]-y_pred_mode_nrn,experiment.data[tr]['Y'][nrn]-y_pred_mode_nrn)
            y_pred_mode_tr.append(y_pred_mode_nrn)
        y_pred_mode_all.append(y_pred_mode_tr)
    y_pred_mode = np.asarray(y_pred_mode_all)
    pred_err_mode = pred_err_mode
    return y_pred_mode, pred_err_mode


def subspaceAngle(F,G):
    '''Partial Translation of the MATLAB code for the article
      - Principal angles between subspaces in an A-based scalar product: algorithms and perturbation estimates
        Knyazev, Andrew V and Argentati, Merico E '''

    EPS = 2e-16
     
    F = np.matrix(np.float64(F))
    G = np.matrix(np.float64(G))

    for i in range(F.shape[1]):
        normi = np.max(F[:,i])
        F[:,i] = F[:,i]/normi

    for i in range(G.shape[1]):
        normi = np.max(G[:,i])
        G[:,i] = G[:,i]/normi

    QF = np.matrix(sp.linalg.orth(F))
    QG = np.matrix(sp.linalg.orth(G))

    q = min(QF.shape[1],QG.shape[1])
    Ys, s, Zs = sp.linalg.svd(QF.T.dot(QG))
    s = np.matrix(np.diag(s))
    if s.shape[0] == 1:
        s = s[0]

    s = np.minimum(np.diag(s),1)
    theta = np.maximum(np.arccos(s),0)
    return max(theta)

def saveVariables(variable, filename):
    f = open(filename, 'wb')
    pickle.dump(variable, f)

def openVariables(filename):
    f = open(filename, 'rb')
    return pickle.load(f)

def approx_jacobian(x,func,epsilon,*args):
    '''Approximate the Jacobian matrix of callable function func

    Parameters:
      * x       - The state vector at which the Jacobian matrix is desired
      * func    - A vector-valued function of the form f(x,*args)
      * epsilon - The step size used to determine the partial derivatives. Set to None to select 
                  the optimal step size. 
      * *args   - Additional arguments passed to func

    Returns:
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of inputs of func.

    Notes:
        The approximation is done using fourth order central difference method.
    '''
    if np.shape(x) == ():
        n = 1
        x = np.asarray([x])
    else:
        n = len(x)

    method = 'FirstOrderCentralDifference'
    method = 'FourthOrderCentralDifference'
    
    x0 = np.asarray(x)
    f0 = func(x0, *args)
    
    if method == 'FirstOrderCentralDifference':
        jac = np.zeros([len(x0),len(f0)])
        df1 = np.zeros([len(x0),len(f0)])
        df2 = np.zeros([len(x0),len(f0)])
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            df1[i] = func(*((x0+dx/2,)+args))
            df2[i] = func(*((x0-dx/2,)+args))
            jac[i] = (df1[i] - df2[i])/epsilon
            dx[i] = 0.0

    if method == 'FourthOrderCentralDifference':
        epsilon = nd._get_epsilon(x,3,epsilon,n)/2.
        jac = np.zeros([len(x0),len(f0)])
        df1 = np.zeros([len(x0),len(f0)])
        df2 = np.zeros([len(x0),len(f0)])
        df3 = np.zeros([len(x0),len(f0)])
        df4 = np.zeros([len(x0),len(f0)])    
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon[i]
            df1[i] = -func(*((x0+2*dx,)+args))
            df2[i] = 8*func(*((x0+dx,)+args))
            df3[i] = -8*func(*((x0-dx,)+args))
            df4[i] = func(*((x0-2*dx,)+args))
            jac[i] = (df1[i]+df2[i] + df3[i] + df4[i])/(12*dx[i])
            dx[i] = 0.0
    return jac.transpose()

def getCdErrorBars(params, experiment, infRes):
    ydim, xdim = np.shape(params['C'])
    def cost(vecCd): return learning.MStepObservationCost(vecCd, xdim, ydim, experiment, infRes)
    def grad(vecCd): return learning.MStepObservationCost_grad(vecCd, xdim, ydim, experiment, infRes)
    
    vecCd = CdtoVecCd(params['C'],params['d'])

    hess = nd.Jacobian(grad)
    hessCd = (hess(vecCd))
    invHessCd = np.linalg.inv(hessCd)#*sigma_C**2*(xdim*ydim+ydim)
    errorBars = np.sqrt(np.diag(invHessCd))
    return errorBars

def seenTrials(experiment, seenIdx):
    seenTrials = []
    seenIdx = np.asarray(seenIdx).flatten()
    for idx in seenIdx:
        seenTrials.append(experiment.data[idx])
    seenExperiment = copy.copy(experiment)
    seenExperiment.data = seenTrials
    seenExperiment.numTrials = len(seenTrials)
    return seenExperiment

def subsampleTrials(experiment, batchSize): 
    '''
    Used for online EM. 
    '''
    numTrials = len(experiment.data)
    # batchTrIdx = np.random.randint(numTrials, size = batchSize)
    batchTrIdx = np.random.choice(numTrials, batchSize, replace=False)
    newTrials = []
    for idx in batchTrIdx:
        newTrials.append(experiment.data[idx])
    newExperiment = copy.copy(experiment)
    newExperiment.data = newTrials
    newExperiment.numTrials = batchSize
    newExperiment.batchTrIdx = batchTrIdx
    return newExperiment

def mvnpdf(x,mean,cov):
    k = len(x)
    xmm = x - mean
    # px = (((2*np.pi)**(-k/2)) * np.linalg.det(cov)**(-1/2)) * np.exp(-1/2*np.dot(xmm.T,np.dot(np.linalg.inv(cov),xmm)))
    px = (2*np.pi)**(-k/2) * np.linalg.det(cov)**(-1/2) * np.exp(-1/2*xmm.T.dot(np.linalg.inv(cov).dot(xmm)))
    return px

def mvnpdf_use_inv_cov(x,mean,invcov):
    k = len(x)
    xmm = x - mean
    # px = (((2*np.pi)**(-k/2)) * np.linalg.det(cov)**(-1/2)) * np.exp(-1/2*np.dot(xmm.T,np.dot(np.linalg.inv(cov),xmm)))
    px = (2*np.pi)**(-k/2) * np.linalg.det(invcov)**(1/2) * np.exp(-1/2*xmm.T.dot(invcov.dot(xmm)))
    return px


#Homemade version of matlab tic and toc functions
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print('Elapsed time is ' + str(time.time() - startTime_for_tictoc) + ' seconds.')
    else:
        print('Toc: start time not set')

# def getAvgFR(experiment):


def initializeParams(xdim, ydim, experiment = None):
    '''Initializes Poisson-GPFA model parameters.
    
    Parameters:
    ===========
      * xdim       : int, latent dimensionality to fit
      * ydim       : int, number of neurons in the dataset
      * experiment : (optional) If a third optional argument of util.dataset object is given, 
                     the fucntion returns a dictionary of parameters obtained by performing Poisson-
                     PCA Leave this argument empty to initialize randomly.
    Returns:
    ========
         A dictionary of model parameters.
    '''
    if experiment == None:
        print('Initializing parameters randomly..')
        params = {
            'C': np.random.rand(ydim,xdim)*2 - 1,
            'd': np.random.randn(ydim)*2 - 2,
            'tau': np.random.rand(xdim)*0.5}   # seconds
    if experiment != None:
        print('Initializing parameters with Poisson-PCA..')
        # make a long raster from all trials called
        spikes = np.zeros([experiment.ydim, experiment.T*experiment.numTrials])
        for tr in range(experiment.numTrials):
            spikes[:,tr*experiment.T:(tr+1)*experiment.T] = experiment.data[tr]['Y']
        # get mean and cov
        meanY = np.mean(spikes,1) + 1e-10
        covY = np.cov(spikes)

        # raise warning if there is a neuron with too few spikes
        # if np.any(meanY/experiment.binSize*1000 < 3):
        #     print('Warning: there is a neuron with a very low firing rate.')

        # moment conversion between Poisson & Gaussian with exponential nonlinearity
        lamb = np.log(np.abs(covY + np.outer(meanY,meanY) - np.diag(meanY))) - np.log(np.outer(meanY,meanY))
        gamma = np.log(meanY)
        # PCA 
        evals, evecs= np.linalg.eig(lamb)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        # select the first xdim eigenvectors 
        evecs = evecs[:, :xdim]

        initC = evecs
        initd = gamma
        params = {
            'C': initC,
            'd': initd,
            # 'tau': np.linspace(0.1,1,xdim)}   # seconds
            'tau': np.random.rand(xdim)*0.5+0.1}   # seconds
    return params

def CdtoVecCd(C, d):
    '''Given C,d returns vec(C,d).

    Parameters:
    ===========
      * C : numpy array of shape (xdim, ydim), loading matrix
      * d : numpy array of shape (ydim, 1), offset parameter
    Returns:
    ========
      * vecCd : numpy array of shape (xdim*ydim+ydim, 1), vectorized C,d
    '''
    ydim,xdim = np.shape(C)
    matCd = np.concatenate([C.T, np.asarray([d])]).T
    vecCd = np.reshape(matCd.T, xdim*ydim + ydim)
    return vecCd

def vecCdtoCd(vecCd, xdim, ydim):
    '''Given vecCd, xdim, ydim, returns C,d.

    Parameters:
    ===========
      * vecCd : numpy array of shape (xdim*ydim+ydim, 1), vectorized C,d
      * xdim : int, latent dimensionality
      * ydim : int, number of neurons
    
    Returns:
    ========
      * C : numpy array of shape (xdim, ydim), loading matrix
    '''
    matCd = np.reshape(vecCd, [xdim+1, ydim]).T
    C = matCd[:,:xdim]
    d = matCd[:,xdim]
    return C, d

def makeCd_big(params, T):
    C_big = np.kron(params['C'],np.eye(T)).T
    d_big = np.kron(np.ndarray.flatten(params['d']),np.ones(T)).T
    return C_big, d_big

def makeK_big(params, trialDur, binSize, epsNoise = 0.001):
    [ydim,xdim] = np.shape(params['C'])
    epsSignal = 1 - epsNoise
    params['tau'] = np.ndarray.flatten(params['tau'])

    T = range(0,int(trialDur/binSize))
    K = np.zeros([xdim, len(T), len(T)])
    K_big = np.zeros([xdim*len(T), xdim*len(T)])
    
    # Make small K (size TxT) for each xdim
    for xd in range(xdim):
        for i in T:
            for j in T:
                K[xd,i,j] = epsSignal*np.exp(-0.5*((T[i]*binSize-T[j]*binSize)**2/(params['tau'][xd]*1000)**2))
        K[xd] = K[xd] + epsNoise * np.eye(len(T))

    # Make big K
    for xd in range(xdim):
        K_big[xd*len(T):(xd+1)*len(T),xd*len(T):(xd+1)*len(T)] = K[xd]
    
    return K_big, K
        
class dataset:
    '''Dataset containing multiple trials of population spike counts. A dataset is sampled from the
    Poisson-GPFA model as described by the equations
        
        x ~ GP(0,K(tau))            - (1)
        y ~ Poisson(exp(Cx+d))      - (2)

    Attributes:
    ===========
      * self.xdim : int, latent dimensionality.
      * self.ydim : int, number of neurons.
      * self.data : list of dictionaries
            The nth element of the list is a dictionary containing the data such that
          - self.data[n]['X'] : numpy array of shape (xdim x T)
                The true latent trajectory of trial n
          - self.data[n]['Y'] : numpy array of shape (ydim x T)
                The spike counts of the population of trial n
      * self.trialDur : int
            The duration of each trial in ms. All trials must have the same length.
      * self.binSize : int, the size of the bin in ms.
      * self.T : int, the number of bins in each trial.
      * self.numTrials : int, the number of trials in the dataset.
      * self.seed : int, seed used to generate the dataset.
      * self.dOffset : float
            The elements of the vector d in equation (2) are drawn from U(-2,0) + dOffset.
      * self.drawSameX : bool, if True, all trials have the same latent trajectory.
      * self.avgFR : float
            Population average firing rate in Hz. Returned by the bound method self.getAvgFiringRate.

    Methods:
    ========
      * self.getAvgFiringRate(self) : bound method
            Computes the average firing rate of the population in Hz and returns it as the attribute
            self.avgFR
      * self.plotTrajectory(self, trialToShow) : bound method
        Plots the latent trajectory and the spike counts of trialToShow (int).
      * self.plotParams(self) : bound method, Plots the parameters. 
    '''
    def __init__(self,
        trialDur = 1000,
        binSize = 10,
        drawSameX = False,
        numTrials = 20,
        xdim = 3,
        ydim = 30,
        seed = 12,
        dOffset = -1,
        fixTau = False,
        fixedTau = None,
        params = None,
        model = 'pgpfa'):

        print('+------------- Simulated Dataset Options -------------+')
        Printer.stdout((str(xdim)+' |').rjust(int(55)))
        Printer.stdout('| Dimensionality of Latent State: ')
        sys.stdout.flush()
        print()
        Printer.stdout((str(ydim)+' |').rjust(int(55)))
        Printer.stdout('| Dimensionality of Observed State (# neurons): ')
        sys.stdout.flush()
        print()
        Printer.stdout((str(trialDur)+' |').rjust(int(55)))
        Printer.stdout('| Duration of trials (ms): ')
        sys.stdout.flush()
        print()
        Printer.stdout((str(binSize)+' |').rjust(int(55)))
        Printer.stdout('| Size of bins (ms): ')
        sys.stdout.flush()
        print()
        Printer.stdout((str(numTrials)+' |').rjust(int(55)))
        Printer.stdout('| Number of Trials: ')
        sys.stdout.flush()
        print()
        print('+-----------------------------------------------------+')


        self.trialDur = trialDur    
        self.binSize = binSize      
        self.drawSameX = drawSameX  
        self.numTrials = numTrials  
        self.xdim = xdim            
        self.ydim = ydim            
        self.seed = seed            

        T = range(int(self.trialDur/self.binSize))

        np.random.seed(self.seed)

        if params == None:
            if model == 'pgpfa':
                params = {
                    'C': np.random.rand(self.ydim, self.xdim)-0.5,
                    # 'd': np.random.rand(self.ydim) + dOffset,
                    'd': np.random.rand(self.ydim)*(-2) + dOffset,
                    'tau': np.abs(np.random.rand(self.xdim)) + 0.01}
                if fixTau == True:
                    params['tau'] = fixedTau
            if model == 'gpfa':
                params = {
                    'C': np.random.rand(self.ydim, self.xdim)-0.5,
                    # 'd': np.random.rand(self.ydim) + dOffset,
                    'd': np.random.rand(self.ydim)*(-2) + dOffset,
                    'tau': np.abs(np.random.rand(self.xdim)) + 0.01,
                    'R': 10*np.diag(np.abs(np.random.rand(ydim)))}
                if fixTau == True:
                    params['tau'] = fixedTau

        self.params = params
        epsSignal = 0.999
        epsNoise = 0.001
        K_big, K = makeK_big(params, trialDur, binSize, epsNoise)

        data = []
        if model == 'pgpfa':
        # Draw same X for all trials
            if drawSameX == True:
                X0 = np.reshape(np.random.multivariate_normal(np.zeros(len(T)*xdim),K_big,1),[xdim,len(T)])
                for i in range(numTrials):
                    data.append({
                        'X': X0,
                        'Y': np.random.poisson(lam=np.exp(np.dot(params['C'],X0)+np.transpose(np.kron(np.ones([len(T),1]),params['d']))))})
                    output = 'Sampling trial %d ...'%(i+1)
                    Printer(output)
            # Draw different X for all trials
            else:
                for i in range(numTrials):
                    X = np.reshape(np.random.multivariate_normal(np.zeros(len(T)*xdim),K_big,1),[xdim,len(T)])
                    data.append({
                        'X': X,
                        'Y': np.random.poisson(lam=np.exp(np.dot(params['C'],X)+np.transpose(np.kron(np.ones([len(T),1]),params['d']))))})
                    output = 'Sampling trial %d ...'%(i+1)
                    Printer(output)
        if model == 'gpfa':
            # Draw same X for all trials
            if drawSameX == True:
                X0 = np.reshape(np.random.multivariate_normal(np.zeros(len(T)*xdim),K_big,1),[xdim,len(T)])
                for i in range(numTrials):
                    data.append({
                        'X': X0,
                        'Y': np.random.multivariate_normal(
                            np.dot(params['C'],X0)+np.transpose(np.kron(np.ones([len(T),1]),params['d'])),
                            np.kron(np.ones([len(T),1]),params['R']))})
                    output = 'Sampling trial %d ...'%(i+1)
                    Printer(output)
            # Draw different X for all trials
            else:
                for i in range(numTrials):
                    X = np.reshape(np.random.multivariate_normal(np.zeros(len(T)*xdim),K_big,1),[xdim,len(T)])
                    data.append({
                        'X': X,
                        'Y': np.random.multivariate_normal(
                            np.dot(params['C'],X).flatten()+np.transpose(np.kron(np.ones([len(T),1]),params['d'])).flatten(),
                            np.kron(np.eye(len(T)),params['R'])).reshape([ydim,len(T)])})
                    output = 'Sampling trial %d ...'%(i+1)
                    Printer(output)





        self.T = len(T)
        self.K_big = K_big
        self.data = data
        self.getAvgFiringRate()
        message = '\nAverage firing rate per neuron in this dataset: %.3f Hz.' %np.mean(self.avgFR)
        print(message)

        self.getAllRaster()
        self.getMeanAndVariance()
        self.fitPolynomialToMeanVar()

    def getAllRaster(self):
        all_raster = np.zeros([self.ydim, len(self.data)*self.T])
        for tr in range(len(self.data)):
            all_raster[:,tr*self.T:(tr+1)*self.T] = self.data[tr]['Y']
        self.all_raster = all_raster

    def getMeanAndVariance(self):
        means = np.zeros([self.ydim, self.T*len(self.data)])
        variances = np.zeros([self.ydim, self.T*len(self.data)])
        for tr in range(len(self.data)):
            for yd in range(self.ydim):
                means[yd,tr] = np.mean(self.data[tr]['Y'][yd,:])
                variances[yd,tr] = np.var(self.data[tr]['Y'][yd,:])
        self.means = means
        self.variances = variances

    def fitPolynomialToMeanVar(self):
        means = self.means.flatten()
        variances = self.variances.flatten()
        def func(x,a,b): return a*x**b
        p, cov = op.curve_fit(func, means, variances, maxfev = 100000)
        self.curve_p = p
        self.curve_p_cov = cov

    def plotMeanVsVariance(self):
        fig, ax = plt.subplots(ncols = 1, figsize = (4,4))
        ax.plot(self.means.flatten(), self.variances.flatten(),'.')
        ax.plot(
            np.linspace(1e-2,max(np.max(self.means),np.max(self.variances)),20),
            np.linspace(1e-2,max(np.max(self.means),np.max(self.variances)),20),
            'g',linewidth=1)
        ax.set_xlim([1e-2,max(np.max(self.means),np.max(self.variances))])
        ax.set_ylim([1e-2,max(np.max(self.means),np.max(self.variances))])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Mean Spike Count')
        ax.set_ylabel('Variance of Spike Count')
        if hasattr(self,'curve_p'):
            x = np.linspace(1e-2,max(np.max(self.means),np.max(self.variances)),20)
            y = self.curve_p[0]*x**self.curve_p[1]
            ax.plot(x,y,'r',linewidth=1)
            plt.legend(['Neuron/Trial','x=y','$f(x) = ax^b$\na=%.2f,b=%.2f'%(self.curve_p[0],self.curve_p[1])],
                frameon = False, framealpha = 0, fontsize = 10,loc = 'best')
        plt.tight_layout()
        ax.grid(which='major')
        simpleaxis(ax)


    def getAvgFiringRate(self):
        avgFR = np.zeros(self.ydim)
        totalSpkCt = 0
        for i in range(self.numTrials):
            avgFR = avgFR + np.sum(self.data[i]['Y'],1)
            totalSpkCt += np.sum(self.data[i]['Y'])
        avgFR = avgFR/self.numTrials/(self.trialDur/1000)
        self.avgFR = avgFR
        self.totalSpkCt = totalSpkCt

    

    def plotTrajectory(self, trialToShow = 0):
        '''
        Plots ground truth latent trajectory, spike counts, parameters
        '''
        fig1, (ax0,ax1) = plt.subplots(nrows = 2, sharex = True, figsize = (5,4))
        raster = ax0.imshow(self.data[trialToShow]['Y'], 
            interpolation = "nearest", aspect = 'auto', cmap = 'gray_r')
        # raster.set_cmap('spectral')
        ax0.set_ylabel('Neuron Index')
        ax0.set_title('Binned Spike Counts')
        ax1.plot(range(self.T),self.data[trialToShow]['X'].T,linewidth=2)
        ax1.set_xlabel('Time ('+str(self.T)+' ms bins)')
        ax1.set_title('Ground Truth Latent Trajectory')
        ax1.set_xlim([0,self.T])
        ax1.grid(which='both')
        plt.tight_layout()

    def plotParams(self): 
        gs = gridspec.GridSpec(2,2)

        ax_C = plt.subplot(gs[0,0])
        ax_d = plt.subplot(gs[1,0])
        ax_K = plt.subplot(gs[:,1])

        ax_C.imshow(self.datasetDetails['params']['C'].T, interpolation = "nearest")
        ax_C.set_title('$C_{true}$')
        ax_C.set_ylabel('Latent Dimension Index')
        ax_C.set_xlabel('Neuron Index')
        ax_C.set_yticks(range(self.xdim))
        ax_d.plot(self.datasetDetails['params']['d'].T)
        ax_d.set_title('$d_{true}$')
        ax_d.set_xlabel('Neuron Index')
        ax_K.imshow(self.K_big, interpolation = "nearest")
        ax_K.set_title('$K_{tau_{true}}$')
        plt.tight_layout()

class MATLABdataset():
    def __init__(self,datfilename, paramfilename = None):
        dataPPGPFA = sio.loadmat(datfilename)

        data = []
        ydim, T = np.shape(dataPPGPFA['dataPPGPFA'][0,0]['spkcount'])
        # xdim, trash = np.shape(dataPPGPFA['dataPPGPFA'][0,0]['x'])
        trialDur = int(dataPPGPFA['dataPPGPFA'][0,0]['trialDur']*1000)
        binSize = int(trialDur/T)

        numTrials = len(dataPPGPFA['dataPPGPFA'].T)

        for i in range(numTrials):
            data.append({
                'Y':dataPPGPFA['dataPPGPFA'][0,i]['spkcount']})
                # 'X':dataPPGPFA['dataPPGPFA'][0,i]['x']})

        self.data = data
        self.ydim = ydim
        # self.xdim = xdim
        self.T = T
        self.trialDur = trialDur
        self.binSize = binSize
        self.numTrials = numTrials

        if paramfilename != None:
            importedParams = sio.loadmat(paramfilename)
            initParams =  importedParams['initParams']
            tau = np.ndarray.flatten(initParams['tau'][0][0])
            C = initParams['C'][0][0]
            d = np.ndarray.flatten(initParams['d'][0][0])
            self.initParams = {'tau': tau, 'C':C, 'd':d}