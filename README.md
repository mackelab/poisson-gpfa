# Poisson-GPFA

This package is written by:
 * Hooram Nam, `hooramnam@openmailbox.org`
 * Jakob Macke, `jakob.macke@caesar.de`

This repository contains different methods for the Gaussian process model with Poisson observations. It has been developed and implemented with the goal of modelling spike-train recordings from neural populations, but some of the methods will be applicable more generally.

In particular, the repository includes methods for

 * Laplace approximation for state-inference
 * Variational method for state-inference
 * Expectation maximisation for parameter learning, using Laplace or Variational inference
     * Full EM, where all available trials are processed in each iteration
     * Variants of stochastic EM, where a subset of avilable trials are processed in each iteration

## Requirements

* matplotlib == 1.4.3
* pandas == 0.16.2
* numpy == 1.10.1
* scipy == 0.16.1
* statsmodels == 0.6.1

## Usage

To get started, run the example script either by `python example.py` in bash or`run example.py` in iPython. The software is developed within the Anaconda python 3 environment. 

If you notice a bug, want to request a feature, or have a question or feedback, please make use of the issue-tracking capabilities of the repository. We love to hear from people using our code -- please send an email to info@mackelab.org.

The code in this repository is a work in progress. This work is published under the GNU General Public License. The code is provided "as is" and has no warranty whatsoever.


