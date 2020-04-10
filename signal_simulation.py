# Copyright (c) 2016, Diarmaid O Cualain; Marijn Stam
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of ecg_simulation nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fourier
import scipy.linalg
import numpy as np
from termcolor import colored
import sys
from random import seed
from random import randint
import pandas as pd

from filters import Filters
from signal_tools import SignalTools



sample_rate = 125   # sample rate, Hz

#NOTE
#The data in this CSV is sampled at a sample rate of 125Hz
#Respiratory rate in this data = 20/min
def importCSV(filename, num_of_breaths, plot=False):

    """
    Imports respiratory data from a CSV generated by the ProtoCentral app: https://github.com/Protocentral/ADS1292rShield_Breakout. 

    Parameters
    ----------
    filename : `string` \n
        Filename of the CSV file to be read
    num_of_breaths : `int` \n
        Number of respiratory cycles to be extracted from the CSV (1 = once in and out)
    plot : `Bool` \n
        Plot the raw data if True, defaults to False.

    Returns
    ----------
    result : `array_like`\n
        Array of size `num_breaths * 375` which holds normalized respiratory data (i.e. y-axis from 0 to 1.)

    Notes
    ----------
    This function is hard-coded to a sample rate of 125Hz and respiratory rate of 20 cycles per minute\n 
    This is the default data our patient monitor outputs. 
    """


    """
    Process CSV to read the raw values, modify "ECG" to column name which holds the data
    """
    data = pd.read_csv(filename)

    resp_data = data['ECG']
    np_resp = np.array(resp_data)


    """
    Normalize the CSV data to 0-1 over the y-axis
    """
    normalized_resp = np.zeros(len(np_resp))
    min_resp = min(np_resp)
    max_resp = max(np_resp)
           

    for idx, value in enumerate(np_resp):
        normalized_resp[idx] = (np_resp[idx] - min_resp) / (max_resp - min_resp)

    """
    Plot the data and slice to amount of breaths. Note that these numbers are static with a sample rate of 125.
    """
    if(plot):
        plt.figure("CSV Data")
        plt.title("Ademhalingssignaal van patiëntsimulator")
        plt.xlabel("Sample")
        plt.ylabel("Normalized amplitude")
        plt.plot(normalized_resp)
        plt.show()
    result = normalized_resp[0:num_of_breaths*375]
    return result


def main():

    seed(1)
    num_of_breaths = 5
    capture_length = num_of_breaths * 3
    # resp_data = importCSV(filename='./data/walk3.csv', num_of_breaths=num_of_breaths, plot=True)
    num_samples = sample_rate * capture_length


    """
    Class instantiation
    Filters gives us access to several LTI or non-LTI filters. LTI filters can be IIR or FIR.
    SignalTools gives us extra tools like sine wave generation, FFT's, downsampling etc.
    """
    filterInterface = Filters(sample_rate, capture_length)
    signalInterface = SignalTools(sample_rate, capture_length)


    """
    Generation of sine waves
    """
    sine_respiratory = signalInterface.sine_generator(5)
    sine_mains = signalInterface.sine_generator(50, 0.1)
    sine_gen = signalInterface.sine_generator(0.4, 0.5)
    sine_gen_2 = signalInterface.sine_generator(0.2, 0.6)




    """
    Add random (gaussian distributed) noise 
    1 in every 10 samples (statistically), 20 samples of noise are added. This simulates erradic movement.
    """
    # noise = np.random.normal(0, 0.2, num_samples)
    # for i, value in enumerate(nk_respiratory):
    #     if(randint(0,10) > 9):
    #         nk_respiratory[i:i+20:1] = value + noise[i:i+20:1]
    # 1 in every 10 samples (statistically), 20 samples of noise are added. This simulates erradic movement.



    """
    Input signals
    """
    # respiratory_noisy =  (nk_respiratory + sine_mains) 


    impulse = signal.unit_impulse(num_samples)


    """
    Downscaling
    """
    # downsample_factor = 5

    # resp_data_lo = signalInterface.decimate(sine_respiratory, downsample_factor)
    # resp_data_lo2 = signalInterface.downsample(sine_respiratory, downsample_factor)
    # signalInterfaceLowRes = SignalTools(sample_rate//downsample_factor, capture_length)
    # filterInterfaceLowRes = Filters(sample_rate//downsample_factor, capture_length)

    
    """
    Applying filters or FFT's
    """

 



    """
    PLAYGROUND
    """




    print(signalInterface.advanced_count(sine_gen))
    # print(signalInterface.original_count(resp_data))


    print(colored('\nDone', 'green'))

    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(colored('\nDone, interrupted', 'green'))    
        sys.exit(0)

#TODO Write tests!