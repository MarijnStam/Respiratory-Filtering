# Copyright (c) 2016, Marijn Stam
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

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import warnings
from termcolor import colored
import signal_tools


class Filters:
    """
    The Filters class contains a set of filter which can be accessed through
    an object of this class. 

    Parameters
    ----------
    sample_rate : int, float
        Sampling rate to use with the functions.

    Notes
    -----
    Functions you call on this class will inherit the sampling rate which you
    have passed to the constructor when instantiating this class.

    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    

    def __init__(self, sampling_rate, capture_length):
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
        self.signalInterface = signal_tools.SignalTools(sampling_rate, capture_length)


    def low_pass(self, data, cutoff, order):
        """
        Low-pass filter
        A low-pass (in this case a Butterworth) filter, passes "all" frequencies below a given cuttoff frequency and filters the 
        frequencies above this cutoff. The order defines the steepness of the cutoff. 
        Useful for filtering recurrent noise at high frequency rates or to prevent aliasing. https://en.wikipedia.org/wiki/Butterworth_filter

        Parameters
        ----------
        data : array_like       
            The array to be filtered
        cutoff : int, float     
            Desired cutoff frequency
        order   : int
            Order of the filter

        Returns
        ----------
        filtered_data : dict
            Filtered signal as "filtered_data", filter characteristics as "sos", filter name as "filter_name" and cutoff as "cutoff"
        """
        normal_cutoff = cutoff / self.nyquist_freq

        plt.figure("Lowpass filter")
        ax = plt.subplot(2, 1, 1)
        plt.plot(data, label="Before filter", color='r')
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.title("Effect of lowpass filter on the signal")

        sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        filtered_data = signal.sosfiltfilt(sos, data)

        plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
        plt.plot(filtered_data, label="After filter", color='g')
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.text(80000, -1.5, "cutoff = %sHz\norder=%s"%(cutoff, order))

        result = dict(filtered_data=filtered_data, sos=sos, filter_name="Low pass filter", cutoff=cutoff)
        self.show_filter_response(result)
        self.signalInterface.fft_plot(result['filtered_data'], 'FFT on low-passed signal')

        return result


    def high_pass(self, data, cutoff, order):
        """
        A high-pass filter functions as the opposite of a low-pass filter. It passes frequencies above a given cutoff and filters 
        frequencies below this cutoff. This filter is a modification of the Butterworth (low-pass) filter. 

        Parameters
        ----------
        data : array_like       
            The array to be filtered
        cutoff : int, float     
            Desired cutoff frequency
        order   : int
            Order of the filter

        Returns
        ----------
        filtered_data : dict
            Filtered signal as "filtered_data", filter characteristics as "sos", filter name as "filter_name" and cutoff as "cutoff"
        """
        normal_cutoff = cutoff / self.nyquist_freq
        plt.figure("Highpass filter")

        ax = plt.subplot(2, 1, 1)
        plt.plot(data, label="Before filter", color='r')
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.title("Effect of highpass filter on the signal")

        sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
        filtered_data = signal.sosfiltfilt(sos, data)
        plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
        plt.plot(filtered_data, label="After filter", color='g')
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.text(80000, -1.5, "cutoff = %sHz\norder=%s"%(cutoff, order))

        result = dict(filtered_data=filtered_data, sos=sos, filter_name="High pass filter", cutoff=cutoff)
        self.show_filter_response(result)
        self.signalInterface.fft_plot(result['filtered_data'], 'FFT on high-passed signal')

        return result





    """ 
    -----------------------------------------------------------------------------------------------------------------------------
    Median filter 
    Median filters excel in filtering out random noise. It functions by averageing the i-1, i and i+1 values of a given array and 
    replacing i by the found median of 
    those 3 values. In our case, with lots of recurring and periodic noise, the median filter is not very effective.
    @Parameters: 
        data:       The array to be filtered
    @Returns
        Array with median filter applied
    -----------------------------------------------------------------------------------------------------------------------------
    """
    def median_filter(self, data, kernel_size):

        plt.figure("Median filter")

        ax = plt.subplot(2, 1, 1)
        plt.plot(data, label="Before filter", color='r')
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.title("Effect of median filter on the signal")

        filtered_data = signal.medfilt(data, kernel_size=kernel_size)

        ax2 = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
        plt.plot(filtered_data, label="After filter", color='g')
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.text(70000, -1.2, "kernel size = %s"%(kernel_size))
        self.signalInterface.fft_plot(filtered_data, 'FFT on median filtered signal')

        return filtered_data






    def show_filter_response(self, filtered_dict):
        if "sos" not in filtered_dict:
            print(colored("Cannot display filter response on non-linear filter!", 'red'))
            return

        w, h = signal.sosfreqz(filtered_dict["sos"], worN=100000)
        plt.figure("Frequency response")
        plt.plot((self.nyquist_freq / np.pi) * w, abs(h))
        plt.plot([0, self.nyquist_freq], [np.sqrt(0.5), np.sqrt(0.5)],
                '--', label='sqrt(0.5)')
        
        if "cutoff" in filtered_dict:
            plt.axvline(x=filtered_dict["cutoff"], color='green', linestyle='--', label='Cuttoff frequency')


        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlim(right=10, left=0)
        plt.title(filtered_dict["filter_name"])
        print(filtered_dict['sos'])
