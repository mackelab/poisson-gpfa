import funs.util as util
import funs.engine as engine
import matplotlib.pyplot as plt
import numpy as np

# Initialize random number generator
np.random.seed(123)

# Specify dataset & fitting parameters
xdim 	  = 2		
ydim      = 20		
numTrials = 5		
trialDur  = 1000 # in ms
binSize   = 20	 # in ms
maxEMiter = 100		
dOffset   = 1	 # controls firing rate

# Sample from the model (make a toy dataset)
training_set  = util.dataset(
	seed	  = np.random.randint(10000),
	xdim 	  = xdim,
	ydim 	  = ydim,
	numTrials = numTrials,
	trialDur  = trialDur,
	binSize   = binSize,
	dOffset   = dOffset,
	fixTau 	  = True, 
	fixedTau  = np.linspace(0.1,0.5,xdim),
	drawSameX = True)

# Initialize parameters using Poisson-PCA
initParams = util.initializeParams(xdim, ydim, training_set)

# Fit using vanilla (batch) EM
fitBatch = engine.PPGPFAfit(
	experiment 		= training_set,
	initParams 		= initParams,
	inferenceMethod = 'laplace',
	EMmode 			= 'Batch',
	maxEMiter 		= maxEMiter)

# Fit using online EM
fitOnline = engine.PPGPFAfit(
 	experiment 		= training_set,
 	initParams 		= initParams,
 	EMmode 			= 'Online',
 	maxEMiter 		= maxEMiter,
 	inferenceMethod = 'laplace',
	batchSize 		= 5)

# Make plots
training_set.plotTrajectory();plt.show()
fitBatch.plotParamSeq()
fitOnline.plotParamSeq();plt.show()

fitBatch.plotTrajectory()
fitOnline.plotTrajectory();plt.show()
