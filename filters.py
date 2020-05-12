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
import traceback

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Filters:
    """
    The Filters class contains a set of filter which can be accessed through
    an object of this class. 

    Parameters
    ----------
    sample_rate : `int`, `float`\n
        Sampling rate to use with the functions.
    capture_length : `int`, `float`\n
        Duration of signal 
    

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


    def lowpass(self, data, cutoff, order, ftype, plot=False):
        """
        Low-pass filter
        IIR LTI Filter: A low-pass (in this case a Butterworth) filter, passes "all" frequencies below a given cuttoff frequency and filters the 
        frequencies above this cutoff. The order defines the steepness of the cutoff. 
        Useful for filtering recurrent noise at high frequency rates or to prevent aliasing. https://en.wikipedia.org/wiki/Butterworth_filter
        
        Parameters
        ----------
        data : `array_like`\n       
            The array to be filtered
        cutoff : `int, float`\n     
            Desired cutoff frequency
        order   : `int`\n
            Order of the filter
        ftype : `string`\n
            Filter type, must be either IIR or FIR
        plot : `bool`\n
            True if you want to plot filter characteristics and result, defaults to False

        Returns
        ----------
        result : `AttrDict`\n
            result.data          : The output signal from the filter\n
            result.sos           : The filter coefficients in Second Order Section form (Only if ftype = IIR)\n
            result.name          : Name of the filter\n
            result.cutoff        : Cutoff frequency used in the filter \n
            result.b             : Numerator of the filter polynomial \n
            result.a             : Denominator of the filter polynomial\n
            result.ftype         : Classification of the filter (IIR or FIR)
            
        """
        normal_cutoff = cutoff / self.nyquist_freq

        if(ftype == "IIR"):
            sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False, output='ba')
            filtered_data = signal.sosfiltfilt(sos, data)
            result = AttrDict(data=filtered_data, sos=sos, name="Lowpass filter", cutoff=cutoff, b=b, a=a, ftype="IIR")

        elif(ftype == "FIR"):
            b = signal.firwin(numtaps=order+1, cutoff=normal_cutoff)
            a = 1.0   #Denominator in an FIR system is 1
            filtered_data = signal.lfilter(b, a, data)
            result = AttrDict(data=filtered_data, name="Lowpass filter", cutoff=cutoff, b=b, a=a, ftype="FIR")

        else:
            print(colored("Filter type must be either IIR or FIR", 'red'))
            return 

        if(plot):
            plt.figure("Lowpass filter")
            plt.grid()
            ax = plt.subplot(2, 1, 1)
            plt.plot(data, label="Voor filter", color='r')
            plt.ylabel("Amplitude")
            plt.legend(loc='upper right')
            plt.title("Effect van een low-pass filter %s" %(ftype))

            plt.subplot(2, 1, 2)
            plt.plot(filtered_data, label="Na filter", color='g')
            plt.xlabel("Sample #")
            plt.ylabel("Amplitude")
            plt.ylim=1
            plt.legend(loc="upper right")
            plt.text(1000, 0.04, "cutoff = %sHz\norder=%s"%(cutoff, order))
            self.show_filter_response(result)
            self.signalInterface.fft_plot(result.data)

    
        return result


    def highpass(self, data, cutoff, order, ftype, plot=False):
        """
        A high-pass filter functions as the opposite of a low-pass filter. It passes frequencies above a given cutoff and filters 
        frequencies below this cutoff. This filter is a modification of the Butterworth (low-pass) filter. 

        Parameters
        ----------
        data : `array_like`\n       
            The array to be filtered
        cutoff : `int, float`\n     
            Desired cutoff frequency
        order   : `int`\n
            Order of the filter
        ftype   : `string`\n
            Type of the filter, must be either FIR or IIR
        plot : `bool`
            True if you want to plot filter characteristics and result, defaults to False

        Returns
        ----------
        result : `AttrDict`\n
            result.data          : The output signal from the filter\n
            result.sos           : The filter coefficients in Second Order Section form \n
            result.name          : Name of the filter\n
            result.cutoff        : Cutoff frequency used in the filter \n
            result.b             : Numerator of the filter polynomial \n
            result.a             : Denominator of the filter polynomial \n
            result.ftype         : Classification of the filter (IIR or FIR)
            
        """
        normal_cutoff = cutoff / self.nyquist_freq
        if(ftype == "IIR"):
            sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False, output='ba')
            filtered_data = signal.sosfiltfilt(sos, data)
            result = AttrDict(data=filtered_data, sos=sos, name="Highpass filter", cutoff=cutoff, b=b, a=a, ftype="IIR")

        elif(ftype == "FIR"):
            b = signal.firwin(numtaps=order+1, cutoff=normal_cutoff, pass_zero='highpass')
            a = 1.0   #Denominator in an FIR system is 1
            filtered_data = signal.lfilter(b, a, data)
            result = AttrDict(data=filtered_data, name="Highpass filter", cutoff=cutoff, b=b, a=a, ftype="FIR")

        else:
            print(colored("Filter type must be either IIR or FIR", 'red'))
            return 

        if(plot):
            plt.figure("Highpass filter")
            plt.grid()

            ax = plt.subplot(2, 1, 1)
            plt.plot(data, label="Before filter", color='r')
            plt.ylabel("Amplitude")
            plt.legend(loc='upper right')
            plt.title("Effect van een high-pass filter %s" %(ftype))


            plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
            plt.plot(filtered_data, label="After filter", color='g')
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper right")
            plt.text(80000, -0.7, "cutoff = %sHz\norder=%s"%(cutoff, order))
            self.show_filter_response(result)
            self.signalInterface.fft_plot(result.data)

        return result


    def median(self, data, kernel_size, plot=False):
        """
        A high-pass filter functions as the opposite of a low-pass filter. It passes frequencies above a given cutoff and filters 
        frequencies below this cutoff. This filter is a modification of the Butterworth (low-pass) filter. 

        Parameters
        ----------
        data    : `array_like`\n       
            The array to be filtered
        kernel_size : `int`\n     
            Must be odd! Kernel window which will be used to calculate averagee around the current value
        plot : `bool`\n
            True if you want to plot filter characteristics and result, defaults to False

        Returns
        ----------
        result : `AttrDict`\n
            result.data          : The output signal from the filter\n
            result.name          : Name of the filter\n
            result.kernel_size   : Size of the kernel window \n
            
        """
        if(kernel_size%2==0):
            print(colored("Median filter kernel size must be odd!\n", 'red'))
            return

        filtered_data = signal.medfilt(data, kernel_size=kernel_size)

        if(plot):
            plt.figure("Median filter")
            plt.grid()

            ax = plt.subplot(2, 1, 1)
            plt.plot(data, label="Before filter", color='r')
            plt.ylabel("Amplitude")
            plt.legend(loc='upper right')
            plt.title("Effect of median filter on the signal")

            ax2 = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
            plt.plot(filtered_data, label="After filter", color='g')
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper right")
            plt.text(70000, -0.7, "kernel size = %s"%(kernel_size))
            self.signalInterface.fft_plot(filtered_data)

        
        result = AttrDict(data=filtered_data, name="Median Filter", kernel_size=kernel_size)
        return result



    def bandpass(self, data, lowcut, highcut, order, ftype, plot=False):
        """
        A bandpass filter is a combination of a lowpass and a highpass filter. It has both a lowcut and a highcut, and passes data only between those.

        Parameters
        ----------
        data : `array_like`\n       
            The array to be filtered
        lowcut : `int, float`\n     
            Desired low cutoff frequency
        highcut : `int, float`\n     
            Desired high cutoff frequency
        order   : `int`\n
            Order of the filter
        ftype   : `string`\n
            Type of the filter, must be either FIR or IIR
        plot : `bool`
            True if you want to plot filter characteristics and result, defaults to False

        Returns
        ----------
        result : `AttrDict`\n
            result.data          : The output signal from the filter\n
            result.sos           : The filter coefficients in Second Order Section form \n
            result.name          : Name of the filter\n
            result.cutoff        : Cutoff frequency used in the filter \n
            result.b             : Numerator of the filter polynomial \n
            result.a             : Denominator of the filter polynomial \n
            result.ftype         : Classification of the filter (IIR or FIR)
            
        """

        normal_low = lowcut / self.nyquist_freq
        normal_high = highcut / self.nyquist_freq

        if(ftype == "IIR"):
            sos = signal.butter(order, [normal_low, normal_high], btype='band', analog=False, output='sos')
            b, a = signal.butter(order, [normal_low, normal_high], btype='band', analog=False, output='ba')
            filtered_data = signal.sosfilt(sos, data)
            result = AttrDict(data=filtered_data, sos=sos, name="Bandpass filter", cutoff=[lowcut, highcut], b=b, a=a, ftype="IIR")
        elif(ftype == "IIRFF"):
            sos = signal.butter(order, [normal_low, normal_high], btype='band', analog=False, output='sos')
            b, a = signal.butter(order, [normal_low, normal_high], btype='band', analog=False, output='ba')
            filtered_data = signal.sosfiltfilt(sos, data)
            result = AttrDict(data=filtered_data, sos=sos, name="Bandpass filter", cutoff=[lowcut, highcut], b=b, a=a, ftype="IIR")
        elif(ftype == "FIR"):
            b = signal.firwin(numtaps=order+1, cutoff=[normal_low, normal_high], pass_zero=False)
            a = 1.0   #Denominator in an FIR system is 1
            filtered_data = signal.lfilter(b, a, data)
            result = AttrDict(data=filtered_data, name="Bandpass filter", cutoff=[lowcut, highcut], b=b, a=a, ftype="FIR") 
        else:
            print(colored("Filter type must be either IIR or FIR", 'red'))
            return

        if(plot):
            plt.figure("Bandpass filter")
            plt.grid()

            ax = plt.subplot(2, 1, 1)
            plt.plot(data, label="Voor filter", color='r')
            plt.ylabel("Amplitude")
            plt.legend(loc='upper right')
            plt.title("Effect van een bandpass filter op het signal")


            plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
            plt.plot(filtered_data, label="Na filter", color='g')
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper right")
            # plt.text(80000, -0.7, "cutoff = %sHz\norder=%s"%(cutoff, order))
            self.show_filter_response(result)
            self.signalInterface.fft_plot(result.data)

        return result

    def show_filter_response(self, filtered_dict):
        """
        Shows the frequency response of a given LTI filter. 

        Parameters
        ----------
        filtered_dict : 'dict' \n
        Pass the dict that was returned from a filter function. 

        Returns
        ----------
        none
        """
        plt.figure("Frequency response")

        if(filtered_dict.ftype=="IIR"):
            w, h = signal.sosfreqz(filtered_dict.sos)
            plt.title("%s IIR" %(filtered_dict.name))
        elif(filtered_dict.ftype=="FIR"):
            w, h = signal.freqz(filtered_dict.b)
            plt.title("%s FIR" %(filtered_dict.name))
        else:
            print(colored("Cannot show frequency respons of non-LTI filter!"), 'red')

        plt.plot((self.nyquist_freq / np.pi) * w, abs(h), label="IIR")
        plt.plot([0, self.nyquist_freq], [np.sqrt(0.5), np.sqrt(0.5)],
                '--', label='-3dB')
        

        if "cutoff" in filtered_dict:
            if(type(filtered_dict.cutoff) != int and type(filtered_dict.cutoff) != float):
                for i in filtered_dict.cutoff:
                    plt.axvline(x=i, color='green', linestyle='--', label='Cuttoff=%.2fHz' %(i))      #Support multiple cutoffs for bandpass
            else:
                plt.axvline(x=filtered_dict.cutoff, color='green', linestyle='--', label='Cuttoff frequentie')
        plt.xlabel('Frequentie (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlim(left=0, right=15)
        

