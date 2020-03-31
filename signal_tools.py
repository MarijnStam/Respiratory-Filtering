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
import scipy.fft as fourier
import scipy.signal as signal
from itertools import islice
from termcolor import colored
import filters
import time

#Fourier Transforms

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SignalTools:
    
    """
    The SignalTools functions as an interface to several utility functions like 
    fft or sine generation. The reason this is defined in a class is to ensure
    several wave-related variables to be identical over all the functions.

    Parameters
    ----------
    sample_rate : `int`, `float` 
        Sampling rate to use with the functions.\n
    capture_length : `int`
        Time or duration of signal.\n

    Notes
    -----
    Functions you call on this class will inherit the sampling rate and capture length which you
    have passed to the constructor when instantiating this class.

    """
    def __init__(self, sample_rate, capture_length):

        self.sample_rate = sample_rate
        self.capture_length = capture_length
        self.num_samples = capture_length * sample_rate


    def fft_plot(self, data):
        """
        Plot the fast-fourier transform of a passed array, use this to analyze
        the frequency-domain of your signals.

        Parameters
        ----------
        data : `array-like`
            Signal array to be analyzed.\n

        Returns
        ---------- 
        result : `AttrDict`\n
            result.x : Linear space of x-axis for the FFT to be plotted on \n
            result.y : Modulus of each frequency bin in the FFT, use this as y-axis \n
            result.freq : Frequency resolution of the FFT \n

        """

        data_fft = fourier.fft(data)

        N = data_fft.size

        sample_spacing = self.capture_length / N
        #Only interested in positive range of the FFT
        frequency_resolution = self.sample_rate / N
        xf = np.linspace(0, frequency_resolution*(N//2), int(N/2))
        modulus = 2.0/N * np.abs(data_fft[:N//2])
        modulus[0] = 0.0 #DC gain = 0

        #Set the x linear axis space to the amount of frequency bins in the FFT
        #Frequency bins are determined by the frequency resolution, or sampling_rate/N

        #TODO Normalize the FFT results

        plt.figure('Fast Fourier transform')
        plt.grid(True, which="both")
        plt.semilogy(xf, modulus)
        plt.xlim(0,1.0/(2.0*sample_spacing))
        plt.title("FFT")
        
        plt.xlabel('Frequentie (Hz)')
        plt.ylabel('Amplitude')
        plt.show(block=True)

        result = AttrDict(x=xf, y=modulus, freq=frequency_resolution)
        return result

        #TODO Make FFT function return the result rather than plotting in function
        #TODO Include the FFT peaks in the result
        



    def sine_generator(self, sinefreq, amplitude_modifier=1):
        """
        Returns a sine-wave at the passed frequency.

        Parameters
        ----------
        sinefreq : `int`, `float`
            Frequency of the generated sine-wave \n
        amplite_modifier: `float`
            Adjusts the amplitude of the sine wave to be larger (>1) or smaller (<1). Defaults to 1 \n

        Returns
        ----------
        y_sine : `array_like`
            Generated sine-wave\n

        Notes
        ----------
        The length of the generated array is based on the sampling rate and the capture length.
        Both these variables are pre-defined in the instantiation of this class to prevent
        missmatching arrays.
        """
        samples = np.linspace(0.0, self.capture_length, self.num_samples, endpoint=False)
        y_sine = np.sin(2 * np.pi * sinefreq * samples)
        return y_sine * amplitude_modifier



    def downsample(self, data, chunk_size, anti_alias=True):
        """
        Returns an array downsampled by a variable factor. The average over a chunk, which size is defined by chunk_size\n
        is calculated and placed into the downsampled array.

        Parameters
        ----------
        data : `array_like`
            1D array to be downsampled\n
        chunk_size: `int` 
            chunks in which the array will be divided, can also be interpreted as downsample factor\n
            For example, input array of size 100 with a chunk size of 20 will data in an array of size 5 \n
        anti_alias: `bool`
            Defaults to True. Applies a low-pass filter to the signal before downsampling to prevent aliasing.
            Skips this step when False. 

        Returns
        ----------
        downsampled : `array_like`
            The downsampled array.

        Notes
        ----------
        Minimal and maximum values are trimmed off the original data based on the chunk size.\n
        The input array is sorted and the array is trimmed so that the first and last quarter are trimmed off.
        """
        downsampled_rate = self.sample_rate / chunk_size
        nyquist = downsampled_rate/2
        filterInterface = filters.Filters(self.sample_rate, self.capture_length)
        slice_int = chunk_size//3
        min_maxed = np.zeros(slice_int)
        if(anti_alias):
            antialias = filterInterface.lowpass(data, nyquist-0.01, order=3, plot=False)
            it = iter(antialias.data)
        else:
            it = iter(data)
        sliced_data = list(iter(lambda: tuple(islice(it, chunk_size)), ()))
        downsampled = np.zeros(len(sliced_data))


        for idx, value in enumerate(sliced_data):
            to_sort = list(value)
            to_sort.sort()
            min_maxed = to_sort[slice_int:chunk_size-slice_int]
            downsampled[idx] = np.average(min_maxed)

        print('Size of original data buffer: \n', len(data))
        print('Size of downsampled data buffer: \n', len(downsampled))

        return downsampled




    def original_count(self, data):
        """
        Original counting method
        This function implements the original counting method to count respiratory cycles as described in:
         https://link.springer.com/article/10.1007/s10439-007-9428-1
        
        Note that this method will apply a bandpass filter on the signal in range 0.1Hz - 0.5Hz. 
        
        Parameters
        ----------
        data : `array_like`\n       
            The signal from which the frequency is to be extracted.
        Returns
        ----------
        result : `float`\n
            The found frequency in the signal.
        """
        filterInterface = filters.Filters(self.sample_rate, self.capture_length)
        result = filterInterface.bandpass(data, lowcut=0.1, highcut=0.5, order=10, ftype="IIR", plot=False)

        maxima = signal.find_peaks(result.data)
        minima = signal.find_peaks(-result.data)

        ordinates = np.zeros(len(maxima[0]))
        true_maxima = true_minima = resp_cycles = np.zeros(0)

        ordinates = np.zeros(len(maxima[0]))
        for idx, i in enumerate(maxima[0]):
            ordinates[idx] = result.data[i]

        quartile = np.quantile(ordinates, .75)
        Q = 0.2 * quartile


        plt.figure("Frequentie extractie")
        plt.title("Originele count-methode")
        plt.plot(result.data, label='Gefiltered signaal')
        plt.axhline(y=Q, color='green', linestyle='--', label='Threshold')

        for i in maxima:
            for j in i:
                if result.data[j] > Q:
                    true_maxima = np.append(true_maxima, j)
                    plt.plot(j, result.data[j], "ro")


        for k in minima:
            for l in k:
                if result.data[l] < 0:
                    true_minima = np.append(true_minima, l)
                    plt.plot(l, result.data[l], "ro", color="green")

        total_distance = 0
        for idx, i in enumerate(true_maxima):
            if(idx < len(true_maxima)-1):
                count = ((true_maxima[idx] < true_minima) & (true_minima < true_maxima[idx+1])).sum()
                if(count == 1):
                    resp_cycles = np.append(resp_cycles, true_maxima[idx])
                    total_distance = total_distance + (true_maxima[idx+1] - true_maxima[idx])
            else:
                break

        plt.grid()
        plt.legend()
        plt.show()
        mean = total_distance / len(resp_cycles)

        frequency = 1 / (mean / self.sample_rate)
        return frequency

    def advanced_count(self, data):

        filterInterface = filters.Filters(self.sample_rate, self.capture_length)
        result = filterInterface.bandpass(data, lowcut=0.1, highcut=0.5, order=10, ftype="IIR", plot=False)
        maxima = signal.find_peaks(result.data)
        minima = signal.find_peaks(-result.data)    

        true_extrema = vertical_diff = np.zeros(0)

        plt.figure("Frequentie extractie")
        plt.title("Geavanceerde count-methode")
        plt.plot(result.data, label="Gefilterd signaal")
        plt.xlabel("Amplitude")
        plt.ylabel("Sample")

        extrema = np.append(maxima[0], minima[0])  
        extrema = np.delete(extrema, -1)
        extrema.sort()

        for idx, i in enumerate(extrema):
            if(idx < len(extrema)-1):
                vertical_diff = np.append(vertical_diff, np.abs(result.data[i] - result.data[extrema[idx+1]]))
            else:
                break

        quartile = np.quantile(vertical_diff, .75)
        Q = 0.3 * quartile
        plt.axhline(y=Q, color='green', linestyle='--', label='Threshold')    

        for idx, i in enumerate(vertical_diff):
            if i > Q:
                true_extrema = np.append(true_extrema, idx)

        total_distance = 0
        for idx, i in enumerate(true_extrema):
            if(idx < len(true_extrema)-1):
                x = extrema[int(i)]
                plt.plot(x, result.data[x], 'ro')
                total_distance = total_distance + (extrema[int(true_extrema[idx+1])] - x)
            else:
                break

        mean = total_distance / len(true_extrema)
        frequency = 1 / (2 * mean / self.sample_rate)
        plt.grid()
        plt.legend()
        plt.show()

        return(frequency)

