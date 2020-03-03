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

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fourier
from itertools import islice
from termcolor import colored

#Fourier Transforms

class SignalTools:
    """
    The SignalTools functions as an interface to several utility functions like 
    fft or sine generation. The reason this is defined in a class is to ensure
    several wave-related variables to be identical over all the functions.

    Parameters
    ----------
    sample_rate : int, float
        Sampling rate to use with the functions.
    capture_length : int
        Time or duration of signal.

    Notes
    -----
    Functions you call on this class will inherit the sampling rate and capture length which you
    have passed to the constructor when instantiating this class.

    """
    def __init__(self, sample_rate, capture_length):

        self.sample_rate = sample_rate
        self.sample_spacing = 1.0 / sample_rate
        self.capture_length = capture_length
        self.num_samples = capture_length * sample_rate


    def fft_plot(self, data, figure_title):
        """
        Plot the fast-fourier transform of a passed array, use this to analyze
        the frequency-domain of your signals.

        Parameters
        ----------
        data : array-like
            Signal array to be analyzed.
        figure_title : string
            Title of the plot    

        """
        data_fft = fourier.fft(data)

        N = data_fft.size
        xf = np.linspace(0, 1.0/(2.0*self.sample_spacing), int(N/2))

        plt.figure('Fast Fourier transform')
        plt.semilogy(xf, 2.0/N * np.abs(data_fft[:N//2]))
        plt.xlim(0, 70)
        plt.title(figure_title)
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show(block=True)



    def sine_generator(self, sinefreq, amplitude_modifier=1):
        """
        Returns a sine-wave at the passed frequency.

        Parameters
        ----------
        sinefreq : int, float
            Frequency of the generated sine-wave
        amplite_modifier: float 
            Adjusts the amplitude of the sine wave to be larger (>1) or smaller (<1). Defaults to 1

        Returns
        ----------
        y_sine : array_like
            Generated sine-wave

        Notes
        ----------
        The length of the generated array is based on the sampling rate and the capture length.
        Both these variables are pre-defined in the instantiation of this class to prevent
        missmatching arrays.
        """
        x = np.linspace(0.0, self.num_samples*(1.0/self.sample_rate), self.num_samples)
        y_sine = np.sin(sinefreq * 2.0 * np.pi*x)
        return y_sine * amplitude_modifier



    def downsample(self, data, chunk_size):
        """
        Returns an array downsampled by a variable factor. The average over a chunk, which size is defined by chunk_size\n
        is calculated and placed into the downsampled array.

        Parameters
        ----------
        data : `array_like`
            1D array to be downsampled\n
        chunk_size: `int` 
            chunks in which the array will be divided, can also be interpreted as downsample factor\n
            For example, input array of size 100 with a chunk size of 20 will result in an array of size 5

        Returns
        ----------
        downsampled : `array_like`
            The downsampled array.

        Notes
        ----------
        Minimal and maximum values are trimmed off the original data based on the chunk size.\n
        The input array is sorted and the array is trimmed so that the first and last quarter are trimmed off.
        """

        slice_int = chunk_size//3
        min_maxed = np.zeros(slice_int)
        it = iter(data)
        sliced_data = list(iter(lambda: tuple(islice(it, chunk_size)), ()))
        downsampled = np.zeros(len(sliced_data))


        for idx, value in enumerate(sliced_data):
            to_sort = list(value)
            to_sort.sort()
            min_maxed = to_sort[slice_int:chunk_size-slice_int]
            downsampled[idx] = np.average(to_sort)

        print('Size of original data buffer: \n', len(data))
        print('Size of downsampled data buffer: \n', len(downsampled))

        return downsampled

    
