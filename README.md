## Import modules
```python
import funs.util as util
import funs.engine as engine
import matplotlib.pyplot as plt
import numpy as np
```
## Initialize random number generator
```python
np.random.seed(123)
```
## Specify dataset & fitting parameters
```python
xdim 	  = 2		
ydim      = 30		
numTrials = 50		
trialDur  = 200	# in ms
binSize   = 20	# in ms
maxEMiter = 100		
dOffset   = 1	# controls firing rate
```
## Sample from the model (simulate a dataset)
```python
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
```
__Output:__
```
+------------- Simulated Dataset Options -------------+
| Dimensionality of Latent State:                   2 |
| Dimensionality of Observed State (# neurons):    30 |
| Duration of trials (ms):                        200 |
| Size of bins (ms):                               20 |
| Number of Trials:                                50 |
+-----------------------------------------------------+
Sampling trial 50 ...
Average firing rate per neuron in this dataset: 65.003 Hz.
```
## Initialize parameters using Poisson-PCA
```python
initParams = util.initializeParams(xdim, ydim, training_set)
```
## Fit using vanilla (batch) EM
```python
fitBatch = engine.PPGPFAfit(
	experiment 		= training_set,
	initParams 		= initParams,
	inferenceMethod = 'laplace',
	EMmode 			= 'Batch',
	maxEMiter 		= maxEMiter)
```
__Output:__
```
+-------------------- Fit Options --------------------+
| Dimensionality of Latent State:                   2 |
| Dimensionality of Observed State (# neurons):    30 |
| EM mode:                                      Batch |
| Max EM iterations:                              100 |
| Inference Method:                           laplace |
+-----------------------------------------------------+
Iteration: 100 of 100, nPLL: = -224.6181
This dataset is a simulated dataset.
Processing performance against ground truth parameters...
```

## Fit using online EM
```python
fitOnline = engine.PPGPFAfit(
 	experiment 		= training_set,
 	initParams 		= initParams,
 	EMmode 			= 'Online',
 	maxEMiter 		= maxEMiter,
 	inferenceMethod = 'laplace',
	batchSize 		= 5)
```
__Output:__
```
+-------------------- Fit Options --------------------+
| Dimensionality of Latent State:                   2 |
| Dimensionality of Observed State (# neurons):    30 |
| EM mode:                                     Online |
| Max EM iterations:                              100 |
| Inference Method:                           laplace |
| Online Param Update Method:                  `diag` |
| Batch size (trials):                              5 |
+-----------------------------------------------------+
Iteration: 100 of 100, nPLL: = -226.2654
This dataset is a simulated dataset.
Processing performance against ground truth parameters...
```