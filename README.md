# Respiratory-Filtering
A repository which is dedicated to finding proper ways to filter and extract the breathing rate from a noisy respiratory signal

## Context
This script is built (and currently being built) for a graduation research assignment. 
For a respiratory signal to be properly interpreted, noise needs to be filtered out. This repo will be the host
of the test tools written to see how noise can be most-properly filtered without altering the orignal signal.
Along with that, several methods of extracting the breathing method from a signal are implemented.

## Running the code
One can simply run the code from command line by running 

`python signal_simulation.py`

In the signal_simulation.py file, the user can modify the processing on the signal. The user is completely free in choosing the processing done, an example of a simple lowpass filter followed by a counting method is supplied as a comment in this file.

The repository holds 2 auxilary files which include several filters or signal tools. An instantiation of the class is made in signal_simulation to provide an interface to these functions.

### Dependencies
This script is entirely written in Python 3.8, other Python versions have not been tested.

The packages which are needed can be installed with the following command
`pip install -r requirements.txt `
