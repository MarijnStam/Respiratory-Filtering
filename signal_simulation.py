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
import neurokit2 as nk
from termcolor import colored
import sys
from random import seed
from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import pandas as pd

from filters import Filters
from signal_tools import SignalTools


capture_length = 10

#NOTE
#The data in this CSV is sampled at a sample rate of 125Hz
#Respiratory rate in this data = 20/min
def importCSV(filename, num_of_breaths):


    """
    Process CSV to read the raw values, modify "ECG" to column name which holds the data
    """
    ss = StandardScaler()
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
    plt.plot(normalized_resp)
    capture_length = num_of_breaths * 3

    return normalized_resp[0:num_of_breaths*375]

# ---------------------------Global variables--------------------------------------------
# ---------------------------------------------------------------------------------------
def main():

    seed(1)

    resp_data = importCSV(filename='cleandata.csv', num_of_breaths=2)

    # Filter requirements.
    order = 5       # Filter order
    sample_rate = 125   # sample rate, Hz

    # The rest between two hearbeats, heartbeats will be simulated in the form of an ECG to add artifacts to the respiratory signal.
    samples_rest = 10

    # Simulated Beats per minute rate
    # For a health, athletic, person, 60 is resting, 180 is intensive exercising
    bpm = 60
    bps = bpm / 60

    # Bit resolution for simulation of ADC
    adc_bit_resolution = 1024

    # Simulated period of time in seconds that the ecg is captured in



    # ---------------------------Start of all scripting logic--------------------------------
    # ---------------------------------------------------------------------------------------

    # The Filter class holds all our filters, we can pass our data to a filter in this class and we get the filtered result.
    filterInterface = Filters(sample_rate, capture_length)
    signalInterface = SignalTools(sample_rate, capture_length)


    # Create sine waves for respiratory signal (.2Hz) and for mains hum (50Hz)
    # nk_respiratory is a more actual respiratory signal simulation based on 
    # the Neurokit2 library, the respiratory rate is amount of breath cycles per minute.
    # nk_respiratory = nk.rsp_simulate(duration=capture_length, sampling_rate=sample_rate, respiratory_rate=20)
    sine_respiratory = signalInterface.sine_generator(5)
    sine_mains = signalInterface.sine_generator(50, 0.1)
    sine_alias = signalInterface.sine_generator(50)

    num_samples = sample_rate * capture_length


    # Add random (gaussian distributed) noise 
    # noise = np.random.normal(0, 0.2, num_samples)
    # for i, value in enumerate(nk_respiratory):
    #     if(randint(0,10) > 9):
    #         nk_respiratory[i:i+20:1] = value + noise[i:i+20:1]
    # 1 in every 10 samples (statistically), 20 samples of noise are added. This simulates erradic movement.



    """
    Input signals
    """
    # respiratory_noisy =  (nk_respiratory + sine_mains) 
    alias_test = sine_respiratory + sine_alias

    impulse = signal.unit_impulse(num_samples)


    """
    Downscaling
    """
    downsample_factor = 2

    # resp_data_lo = signalInterface.downsample(resp_data, downsample_factor)
    # signalInterfaceLowRes = signal_tools.SignalTools(sample_rate//downsample_factor, capture_length)
    # filterInterfaceLowRes = filters.Filters(sample_rate//downsample_factor, capture_length)

    

    fir = signal.firwin(numtaps=101, cutoff=5, fs=125)

    
    """
    Applying filters or FFT's
    """
    filterInterface.bandpass(resp_data, 1, 5, 10, True)




    """
    Outputs and plotting
    """



    # filterInterfaceLowRes.high_pass(lo_filtered.filtered_data, cutoff=5, order=3, plot=True)
    
    # signalInterfaceLowRes.fft_plot(downsampled_resp, figure_title="FFT on the downsampled result")



    # impulse_response_low = filterInterface.low_pass(impulse, cutoff=20, order=3, plot=True)
    # impulse_response_high = filterInterface.high_pass(impulse, cutoff=20, order=3, plot=True)
    # print(impulse_response_high.filtered_data) #NEVER ZERO, IIR!



    plt.show()

    print(colored('\nDone', 'green'))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(colored('\nDone, interrupted', 'green'))    
        sys.exit(0)

#TODO Write tests!