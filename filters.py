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
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2


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
        filtered_data : array_like
            Filtered signal
        """
        normal_cutoff = cutoff / self.nyquist_freq

        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = signal.filtfilt(b, a, data)

        return filtered_data


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
        filtered_data : array_like
            Filtered signal
        """
        normal_cutoff = cutoff / self.nyquist_freq

        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = signal.filtfilt(b, a, data)

        return filtered_data





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
    def median_filter(self, data):
        return signal.medfilt(data)