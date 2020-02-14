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

#Fourier Transforms

class SignalTools:

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.sample_spacing = 1.0 / sample_rate


    def fft_plot(self, data, figure_title):

        data_fft = fourier.fft(data)

        N = data_fft.size
        xf = np.linspace(0, 1.0/(2.0*self.sample_spacing), int(N/2))

        plt.figure('Fast Fourier transform')
        axes = plt.gca()
        axes.set_xlim([0,100])

        plt.semilogy(xf, 2.0/N * np.abs(data_fft[:N//2]))
        plt.title(figure_title)
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()


##TODO
# TRY FFT OVER SMALLER TIME PERIOD RATHER THAN ENTIRE WAVEFORM
# TRY ADDING ALL THE RESULTING FREQUENCIES IN HZ AS A SUM SINE WAVE
