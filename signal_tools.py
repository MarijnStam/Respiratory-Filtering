# Copyright (c) 2020, Marijn Stam
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of Respiratory-Filtering nor the names of its
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


        max_idx = np.argmax(modulus)                
        peak = max_idx * frequency_resolution
        #Determine the most prevelant frequency
        

        plt.figure('Fast Fourier transform')
        plt.grid(True, which="both")
        plt.semilogy(xf, modulus)
        plt.xlim(0,self.sample_rate/2)
        plt.title("FFT")
        
        plt.xlabel('Frequentie (Hz)')
        plt.ylabel('Amplitude')
        plt.scatter(x=peak, y=np.max(modulus), color='green', label='Piek bij: %0.5sHz' %peak)
        plt.legend()
        plt.show(block=True)

        result = AttrDict(x=xf, y=modulus, freq=frequency_resolution)
        return result



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
            antialias = filterInterface.lowpass(data, nyquist-0.01, order=8, ftype="IIR", plot=False)
            it = iter(antialias.data)
        else:
            it = iter(data)
        #Apply an anti-aliasing filter by default

        sliced_data = list(iter(lambda: tuple(islice(it, chunk_size)), ()))
        downsampled = np.zeros(len(sliced_data))


        for idx, value in enumerate(sliced_data):
            to_sort = list(value)
            to_sort.sort()
            min_maxed = to_sort[slice_int:chunk_size-slice_int]
            downsampled[idx] = np.average(min_maxed)
        #Determine the average of the chunk in the original array and use this as the new value in the downsampled array

        print('Size of original data buffer: \n', len(data))
        print('Size of downsampled data buffer: \n', len(downsampled))

        return downsampled

    def decimate(self, data, factor, anti_alias=True):
        """
        Returns an array which is decimated by a factor. Decimation simply means that out of every M samples, 1 is kept and the rest is discarded,
        M is the factor.

        Parameters
        ----------
        data : `array_like`
            1D array to be decimated\n
        factor: `int` 
            Factor by which the array is decimated.\n
        anti_alias: `bool`
            Defaults to True. Applies a low-pass filter to the signal before decimation to prevent aliasing.
            Skips this step when False. 

        Returns
        ----------
        downsampled : `array_like`
            The decimated array.

        """
        downsampled_rate = self.sample_rate / factor
        nyquist = downsampled_rate/2
        filterInterface = filters.Filters(self.sample_rate, self.capture_length)

        if(anti_alias):
            antialias = filterInterface.lowpass(data, nyquist-0.01, order=8, ftype="IIR", plot=False)
            it = iter(antialias.data)
        else:
            it = iter(data)
        sliced_data = list(iter(lambda: tuple(islice(it, factor)), ()))
        downsampled = np.zeros(len(sliced_data))
        #Apply an anti-aliasing filter by default

        for idx, value in enumerate(sliced_data):
            to_decimate = list(value)
            downsampled[idx] = to_decimate[0]


        print('Size of original data buffer: \n', len(data))
        print('Size of decimated data buffer: \n', len(downsampled))

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
        true_maxima = true_minima = resp_cycles = np.zeros(0)
        filterInterface = filters.Filters(self.sample_rate, self.capture_length)
        result = filterInterface.bandpass(data, lowcut=0.5, highcut=5, order=10, ftype="IIR", plot=True)
        #Apply the filter suggested by the paper.


        maxima = signal.find_peaks(result.data)
        minima = signal.find_peaks(-result.data)
        #Find the extrema of the signal

        ordinates = np.zeros(len(maxima[0]))
        for idx, i in enumerate(maxima[0]):
            ordinates[idx] = result.data[i]
        #Find the ordinate values of all maxima

        quartile = np.quantile(ordinates, .75)
        Q = 0.2 * quartile
        #Define a threshold Q as 0.2 * third quartile of the ordinates

        plt.figure("Frequentie extractie")
        plt.title("Originele count-methode")
        plt.plot(result.data, label='Gefiltered signaal')
        plt.axhline(y=Q, color='green', linestyle='--', label='Threshold')

        for i in maxima:
            for j in i:
                if result.data[j] > Q:
                    true_maxima = np.append(true_maxima, j)
                    plt.plot(j, result.data[j], "ro")
        #True maxima are defined to be a maxima above Q

        for k in minima:
            for l in k:
                if result.data[l] < 0:
                    true_minima = np.append(true_minima, l)
                    plt.plot(l, result.data[l], "ro", color="green")
        #True minima are defined to be a minima below 0

        total_distance = 0
        for idx, i in enumerate(true_maxima):
            if(idx < len(true_maxima)-1):
                count = ((true_maxima[idx] < true_minima) & (true_minima < true_maxima[idx+1])).sum()
                #Find whether a respiratory cycle is present.
                #this is defined to start and end at a true maxima, only and only if there is a single true minima between these two.
                if(count == 1):
                    resp_cycles = np.append(resp_cycles, true_maxima[idx])
                    total_distance = total_distance + (true_maxima[idx+1] - true_maxima[idx])
                #Add the distance of the respiratory cycle to the running total.
            else:
                break
        

        plt.grid()
        plt.legend()
        plt.show()
        mean = total_distance / len(resp_cycles)
        frequency = 1 / (mean / self.sample_rate)
        #Determine the respiratory rate from the total distance and amount of respiratory cycles.

        return frequency

    def advanced_count(self, data):
        """
        Advanced counting method
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

        plt.figure()
        filterInterface = filters.Filters(self.sample_rate, self.capture_length)
        result = filterInterface.bandpass(data, lowcut=0.1, highcut=0.5, order=5, ftype="IIR", plot=False)
        #Apply the filter as suggested by the paper

        
        extrema = []
        y_dif = []

        maxima = signal.find_peaks(result.data)
        minima = signal.find_peaks(-result.data)
        #Find the extrema in the signal    

        maxima_list = list(maxima[0])
        minima_list = list(minima[0])
        #Cast the tuples as returned by the find_peaks() function to lists for further processing
        
        extrema = maxima_list + minima_list
        extrema.sort()
        #Sort all the extrema. This will give a list with all the extrema in sequence.

        def calculate_diff(extrema):
            """
            Calcultate the vertical (ordinate) differences between sequential extrema

            Parameters
            ----------
            extrema : `array_like`\n       
                The list of extrema
            Returns
            ----------
            result : `array_like`\n
                The ordinate differences between the sequential extrema.
            """
            y_dif.clear()
            for idx, i in enumerate(extrema):
                if(idx < len(extrema)-1):
                    y_dif.append((i, np.abs(result.data[i] - result.data[extrema[idx+1]])))

            return y_dif



        def threshold_check(data, threshold):
            """
            Recursive function to minimize the extrema list to only contain pairs with an ordinate difference above the threshold

            Parameters
            ----------
            data : `array_like`\n       
                The list of extrema
            threshold : `float`\n       
                The threshold for a minimum ordinate difference between two extrema.
            Returns
            ----------
            result : `array_like`\n
                The list of extrema, all sequential extrema have an ordinate difference higher than the threshold.
            """
            
            y = [i[1] for i in data]
            min_distance = min(y)
            min_index = y.index(min_distance)
            #Find the smallest ordinate difference between two sequential extrema

            if min_distance < threshold:
                if min_index == (len(data)-1):
                    extrema.remove(data[min_index][0])
                    extrema.pop()
                    
                else:
                    extrema.remove(data[min_index][0])
                    extrema.remove(data[min_index+1][0])
            #If the smallest ordinate difference is smaller than the threshold, remove the pair
            
                new_y = calculate_diff(extrema)
                #As a pair is deleted, new pairs are introduced and new vertical differences need to be calculated
                threshold_check(new_y, threshold)
                #Check the threshold again with the new vertical differences until all is above the threshold
            
            return [i[0] for i in data]

                
        initial_vdiff = calculate_diff(extrema)

        y = [i[1] for i in initial_vdiff]
        quartile = np.quantile(y, .75)
        Q = 0.3 * quartile
        #Define the threshold Q as 0.3 * the third quartile of the vertical differences


        plt.axhline(y=Q, color='green', linestyle='--', label='Threshold')    

        true_extrema = threshold_check(initial_vdiff, Q)
        #Minimize the extrema until each pair of extrema satisfies the vertical difference threshold

        plt.plot(result.data)
        for i in true_extrema:
            plt.plot(i, result.data[i], "ro")
        
        resp_cycles = []
        total_distance = 0
        for idx, i in enumerate(true_extrema):
            if idx < len(true_extrema) - 1:
                total_distance = total_distance + (true_extrema[idx+1] - i)
        #Running total for the absicca difference between extrema
        

        
        if(len(true_extrema) % 2) != 0:
            resp_cycles = len(true_extrema) - 1
        else:
            resp_cycles = len(true_extrema)
        #Determine amount of respiratory cycles

        mean = total_distance / resp_cycles
        frequency = 1 / (2 * (mean) / self.sample_rate)
        #Calculate the respiratory rate from the amount of respiratory cycles and the total distance which these cover.

        return frequency








