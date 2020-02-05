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
import numpy
import filters


def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * numpy.pi * sinefreq
    t_sine = numpy.linspace(0, T, nsamples, endpoint=False)
    y_sine = numpy.sin(w * t_sine)
    y_sine = y_sine * 0.3   #Adjust amplitude
    return y_sine


# Filter requirements.
order = 5       # Filter order
fs = 60       # sample rate, Hz


# The Filter class holds all our filters, we can pass our data to a filter in this class and we get the filtered result.
filterInterface = filters.Filters(fs)


print('Simulating heart ecg')

# The "Daubechies" wavelet is a rough approximation to a real,
# single, heart beat ("pqrst") signal
pqrst = signal.wavelets.daub(10)

# Add the gap after the pqrst when the heart is resting. 
samples_rest = 10
zero_array = numpy.zeros(samples_rest, dtype=float)
pqrst_full = numpy.concatenate([pqrst,zero_array])

# Simulated Beats per minute rate
# For a health, athletic, person, 60 is resting, 180 is intensive exercising
bpm = 60
bps = bpm / 60

# Simumated period of time in seconds that the ecg is captured in
capture_length = 20

# Caculate the number of beats in capture time period 
# Round the number to simplify things
num_heart_beats = int(capture_length * bps)

# Concatonate together the number of heart beats needed
ecg_template = numpy.tile(pqrst_full , num_heart_beats)


# Create sine waves for respistory noise (.2Hz) and for mains hum (50Hz)
sine_respitory = sine_generator(fs, .2, capture_length)
sine_mains = sine_generator(fs, 50, capture_length)

# Add random (gaussian distributed) noise 
noise = numpy.random.normal(0, 0.05, len(ecg_template))
ecg_template_noisy = noise + ecg_template 


# Simulate an ADC by sampling the noisy ecg template to produce the values
# Might be worth checking nyquist here 
# e.g. sampling rate >= (2 * template sampling rate)

num_samples = fs * capture_length

ecg_sampled = signal.resample(ecg_template_noisy, int(num_samples))
mains_sampled = signal.resample(sine_mains, int(num_samples))
respitory_sampled = signal.resample(sine_respitory, int(num_samples))

# Scale the normalised amplitude of the sampled ecg to whatever the ADC 
# bit resolution is
# note: check if this is correct: not sure if there should be negative bit values. 
adc_bit_resolution = 1024
ecg =  adc_bit_resolution * ecg_sampled
ecg_with_resp =  (ecg_sampled + respitory_sampled + mains_sampled) * adc_bit_resolution
ecg_filtered_high = filterInterface.highPass(ecg_with_resp, cutoff=0.3, order=5)

ecg_filtered_high_low = filterInterface.lowPass(ecg_filtered_high, cutoff=7, order=5)

# Plot the graphs

plt.figure('ECG Filtering')

plt.subplot(4, 1, 1)
plt.plot(ecg_template)
plt.ylabel('bit value')
plt.title('Original sampled ECG')
plt.xticks(color='w')

plt.subplot(4,1,2)
plt.plot(ecg_with_resp)
plt.ylabel('bit value')
plt.title('Noisy ECG with added 0.2Hz signal and mains hum (50Hz)')
plt.xticks(color='w')

plt.subplot(4,1,3)
plt.plot(ecg_filtered_high)
plt.ylabel('bit value')
plt.title('Highpass filtered signal')
plt.xticks(color='w')

plt.subplot(4,1,4)
plt.plot(ecg_filtered_high_low)
plt.ylabel('bit value')
plt.xlabel('Sample')
plt.title('High and lowpassed filtered signal')
plt.xticks(color='w')



plt.show()

print('Done')
