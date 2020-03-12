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
import numpy as np
import filters
import signal_tools
import neurokit2 as nk
from termcolor import colored
import sys
from random import seed
from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import pandas as pd

#NOTE
#The data in this CSV is sampled at a sample rate of 125Hz
def importCSV(filename, num_of_breaths):
    ss = StandardScaler()
    data = pd.read_csv(filename)

    resp_data = data['Kolom2']
    # reshaped = resp_data.values.reshape(-1, 1)
    # resp_std = ss.fit_transform(reshaped)
    np_resp = np.array(resp_data)
    plt.plot(np_resp)

    return np_resp[0:1250]

# ---------------------------Global variables--------------------------------------------
# ---------------------------------------------------------------------------------------
def main():

    seed(1)

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
    capture_length = 10


    # ---------------------------Start of all scripting logic--------------------------------
    # ---------------------------------------------------------------------------------------

    # The Filter class holds all our filters, we can pass our data to a filter in this class and we get the filtered result.
    filterInterface = filters.Filters(sample_rate, capture_length)
    signalInterface = signal_tools.SignalTools(sample_rate, capture_length)
    signalInterfacePlot = signal_tools.SignalTools(sample_rate, capture_length=2)




    # The "Daubechies" wavelet is a rough approximation to a real,
    # single, heart beat ("pqrst") signal
    pqrst = signal.wavelets.daub(10)

    # Add the gap after the pqrst when the heart is resting. 
    zero_array = np.zeros(samples_rest, dtype=float)
    pqrst_full = np.concatenate([pqrst,zero_array])

    # Caculate the number of beats in capture time period 
    # Round the number to simplify things
    num_heart_beats = int(capture_length * bps)

    # Concatonate together the number of heart beats needed
    ecg_template = np.tile(pqrst_full , num_heart_beats)


    # Create sine waves for respiratory signal (.2Hz) and for mains hum (50Hz)
    # nk_respiratory is a more actual respiratory signal simulation based on 
    # the Neurokit2 library, the respiratory rate is amount of breath cycles per minute.
    # nk_respiratory = nk.rsp_simulate(duration=capture_length, sampling_rate=sample_rate, respiratory_rate=20)
    sine_respiratory = signalInterface.sine_generator(5)
    sine_respiratory_tiny = signalInterfacePlot.sine_generator(5)
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
    # resp_data = importCSV(filename='real.csv', num_of_breaths=2)
    impulse = signal.unit_impulse(num_samples, 'mid')


    """
    Downscaling
    """
    downsample_factor = 2

    # resp_data_lo = signalInterface.downsample(resp_data, downsample_factor)
    # signalInterfaceLowRes = signal_tools.SignalTools(sample_rate//downsample_factor, capture_length)
    # filterInterfaceLowRes = filters.Filters(sample_rate//downsample_factor, capture_length)


    """
    Applying filters or FFT's
    """
    lo_impulse = filterInterface.low_pass(impulse, 5, 10)
    fft_result = signalInterface.fft_plot(lo_impulse.filtered_data, figure_title="FFT op impuls respons")


    min_y = min(fft_result.y)
    max_y = max(fft_result.y)                   #NORMALIZE DATA TO 0..1
    norm = np.zeros(len(fft_result.y))

    for idx, value in enumerate(fft_result.y):
        norm[idx] = (fft_result.y[idx] - min_y) / max_y - min_y


    #make some pretty plots
    plt.plot(fft_result.x, norm)
    plt.xlim(0,10)      #WE DID IT REDDIT



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