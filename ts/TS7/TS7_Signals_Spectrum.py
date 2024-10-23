# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:08:17 2024

@author: elosi

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt   
import scipy.io as sio

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

# Lectura de ECG #
fs = 1000 # Hz
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead[5000:12000])
plt.title('ECG')

spectrum = np.fft.fft(ecg_one_lead, axis=0)/N
f = np.fft.fftfreq(N, 1/fs)
periodogram = np.abs(spectrum)**2
freq, welch = sp.signal.welch(ecg_one_lead, fs=fs, nperseg=N/5, axis=0)

# Find the peak power value
Bw_amplitude = np.max(welch)/2
# Find the frequencies where the power is greater than the threshold
significant_indices = np.where(welch >= Bw_amplitude)[0]  # Indices where the PSD is above the threshold
significant_frequencies = freq[significant_indices]
# Bandwidth estimation
bandwidth = significant_frequencies[-1] - significant_frequencies[0]

print(f"3 dB Bandwidth: {bandwidth} Hz")

plt.figure();
plt.semilogy(f[:N//2], periodogram[:N//2])
plt.title('Periodograma')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')

plt.figure();
plt.semilogy(freq, welch)
plt.title('Welch')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')

# Lectura de pletismografía (PPG)  #
fs = 400 # Hz
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
N = len(ppg)

plt.figure()
plt.plot(ppg)
plt.title('PPG')

spectrum = np.fft.fft(ppg, axis=0)/N
f = np.fft.fftfreq(N, 1/fs)
periodogram = np.abs(spectrum)**2
freq, welch = sp.signal.welch(ppg, fs=fs, nperseg=N/5, axis=0)

plt.figure();
plt.semilogy(f[:N//2], periodogram[:N//2])
plt.title('Periodograma')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')

plt.figure();
plt.semilogy(freq, welch)
plt.title('Welch')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')

# Lectura de audio #
fs, wav_data = sio.wavfile.read('prueba psd.wav')

plt.figure()
plt.plot(wav_data)
plt.title('Audio')

spectrum = np.fft.fft(wav_data, axis=0)/N
f = np.fft.fftfreq(N, 1/fs)
periodogram = np.abs(spectrum)**2
freq, welch = sp.signal.welch(wav_data, fs=fs, nperseg=N/5, axis=0)

plt.figure();
plt.semilogy(f[:N//2], periodogram[:N//2])
plt.title('Periodograma')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')

plt.figure();
plt.semilogy(freq, welch)
plt.title('Welch')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')