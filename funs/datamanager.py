import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import funs.util as util
import funs.engine as engine
import pdb

class StevensonDataset():
    def __init__(
        self,
        subject_id=0,
        ydim = 90,
        trialDur = 1400,
        binSize = 10,
        numTrials = 100,
        ydimData = False,
        numTrData = True):

        T = int(trialDur/binSize)
        matdat = sio.loadmat('data/Stevenson_2011_e1.mat')
        self.matdat = matdat
        if numTrData: numTrials = len(matdat['Subject'][subject_id]['Trial'][0])
        if ydimData: ydim = len(matdat['Subject'][subject_id]['Trial'][0][0]['Neuron'][0])
        data = []
        trial_durs = []
        for trial_id in range(numTrials):
            trial_time = np.asarray(matdat['Subject'][subject_id]['Trial'][0][trial_id]['Time'][0]).flatten()
            trial_big_time = np.min(trial_time)
            trial_end_time = np.max(trial_time)
            trial_durs.append(trial_end_time - trial_big_time)
        for trial_id in range(int(numTrials/2),numTrials):
            Y = []
            spike_time = []
            trial_time = np.asarray(matdat['Subject'][subject_id]['Trial'][0][trial_id]['Time'][0]).flatten()
            trial_big_time = np.min(trial_time)
            trial_end_time = trial_big_time + trialDur/1000
            for yd in range(ydim):
                Y.append(np.asarray(np.histogram(matdat['Subject'][subject_id]['Trial'][0][trial_id]['Neuron'][0][yd][0][0].flatten(),
                                      T,range=(trial_big_time,trial_end_time))[0]))
                spkt = np.asarray(matdat['Subject'][subject_id]['Trial'][0][trial_id]['Neuron'][0][yd][0][0].flatten() - trial_big_time)
                spkt = spkt[spkt<trialDur/1000]
                spike_time.append(spkt)
            data.append({'Y':np.asarray(Y),'spike_time':spike_time})

        self.trial_durs = trial_durs    
        self.data = data
        self.trialDur = trialDur
        self.binSize = binSize
        self.numTrials = int(numTrials/2)
        self.ydim = ydim        
        self.T = T
        util.dataset.getMeanAndVariance(self)
        util.dataset.getAvgFiringRate(self)
        util.dataset.getAllRaster(self)



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

        ax_C.imshow(self.params['C'].T, interpolation = "nearest")
        ax_C.set_title('$C_{true}$')
        ax_C.set_ylabel('Latent Dimension Index')
        ax_C.set_xlabel('Neuron Index')
        ax_C.set_yticks(range(self.xdim))
        ax_d.plot(self.params['d'].T)
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