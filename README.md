# Respiratory-Filtering
A repository which is dedicated to finding proper ways to filter a noisy respiratory signal

## Context
This script is built (and currently being built) for a graduation research assignment. 
For a respiratory signal to be properly interpreted, noise needs to be filtered out. This repo will be the host
of the test tools written to see how noise can be most-properly filtered without altering the orignal signal.
Along with that, benchmarking of the filtering will be researched and tested.

## Running the code
One can simply run the code from command line by running 

`python signal_simulation.py`

This will automatically simulate and plot various signals. They have been labeled accordingly and can be adjusted to whatever you wish
Along with this, Fast Fourier Transforms can be called on any of the waveforms, to see which frequencies are most dominant. This can be useful for analyzing noise.

The repository holds 2 auxilary files which include several filters or signal tools. An instantiation of the class in these files can be made to easily interface with the functions included in these.

### Dependencies
This script is entirely written in Python 3.8, other Python versions have not been tested.

The following packages are needed:

* matplotlib.pyplot
* scipy.signal
* scipy.fftpack
* numpy
