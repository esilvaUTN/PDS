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

    return a.reshape(a.shape[0], 1)

# Lectura de ECG #
fs = 1000 # Hz
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
ecg_one_lead = ecg_one_lead[0:10000]
N = len(ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead)
plt.title('ECG')

spectrum = np.fft.fft(ecg_one_lead, axis=0)/N
f = np.fft.fftfreq(N, 1/fs)
periodogram = fs*(np.abs(spectrum)**2)
freq, welch = sp.signal.welch(ecg_one_lead, fs=fs, nperseg=N/5, axis=0)

Pperio = np.sum(periodogram[:N//2])
periodogram = periodogram[:N//2]/Pperio
Paccum_perio = np.cumsum(periodogram)
thresh = 0.99
index = np.where(Paccum_perio >= thresh)[0][0]
pos_f = f[:N//2]
Bw = pos_f[index]
print('Ancho de banda a partir de Periodograma: ' + str(Bw))

Pwelch = np.sum(welch)
welch = welch/Pwelch
Paccum_welch = np.cumsum(welch)
thresh = 0.99
index = np.where(Paccum_welch >= thresh)[0][0]
Bw = freq[index]
print('Ancho de banda a partir de Welch: ' + str(Bw))

plt.figure();
plt.semilogy(pos_f, periodogram, label='Periodograma')
plt.semilogy(freq, welch, label='Welch')
plt.title('PDS')
plt.title('Welch')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend()

f_mask = (f >= 0) & (f <= Bw)
trunc_spectrum = spectrum
trunc_spectrum[~f_mask] = 0
ecg_rebuilt = np.fft.ifft(trunc_spectrum, axis=0)

plt.figure()
plt.plot(np.real(ecg_rebuilt))
plt.grid(True)
plt.title('Señal temporal reconstruida')
plt.show()

# Lectura de pletismografía (PPG)  #
fs = 400 # Hz
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
N = len(ppg)

plt.figure()
plt.plot(ppg)
plt.title('PPG')

spectrum = np.fft.fft(ppg, axis=0)/N
f = np.fft.fftfreq(N, 1/fs)
periodogram = fs*(np.abs(spectrum)**2)
freq, welch = sp.signal.welch(ppg, fs=fs, nperseg=N/5, axis=0)

Pperio = np.sum(periodogram[:N//2])
periodogram = periodogram[:N//2]/Pperio
Paccum_perio = np.cumsum(periodogram)
thresh = 0.99
index = np.where(Paccum_perio >= thresh)[0][0]
pos_f = f[:N//2]
Bw = pos_f[index]
print('Ancho de banda a partir de Periodograma: ' + str(Bw))

Pwelch = np.sum(welch)
welch = welch/Pwelch
Paccum_welch = np.cumsum(welch)
thresh = 0.99
index = np.where(Paccum_welch >= thresh)[0][0]
Bw = freq[index]
print('Ancho de banda a partir de Welch: ' + str(Bw))

plt.figure();
plt.semilogy(pos_f, periodogram, label='Periodograma')
plt.semilogy(freq, welch, label='Welch')
plt.title('PDS')
plt.title('Welch')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend()

f_mask = (f >= 0) & (f <= Bw)
trunc_spectrum = spectrum
trunc_spectrum[~f_mask] = 0
ecg_rebuilt = np.fft.ifft(trunc_spectrum, axis=0)

plt.figure()
plt.plot(np.real(ecg_rebuilt))
plt.grid(True)
plt.title('Señal temporal reconstruida')
plt.show()


# Lectura de audio #
fs, wav_data = sio.wavfile.read('silbido.wav')
N = len(wav_data)

plt.figure()
plt.plot(wav_data)
plt.title('Audio')

spectrum = np.fft.fft(wav_data, axis=0)/N
f = np.fft.fftfreq(N, 1/fs)
periodogram = fs*(np.abs(spectrum)**2)
freq, welch = sp.signal.welch(wav_data, fs=fs, nperseg=N/5, axis=0)

Pperio = np.sum(periodogram[:N//2])
periodogram = periodogram[:N//2]/Pperio
Paccum_perio = np.cumsum(periodogram)
thresh = 0.99
index = np.where(Paccum_perio >= thresh)[0][0]
pos_f = f[:N//2]
Bw = pos_f[index]
print('Ancho de banda a partir de Periodograma: ' + str(Bw))

Pwelch = np.sum(welch)
welch = welch/Pwelch
Paccum_welch = np.cumsum(welch)
thresh = 0.99
index = np.where(Paccum_welch >= thresh)[0][0]
Bw = freq[index]
print('Ancho de banda a partir de Welch: ' + str(Bw))

plt.figure();
plt.semilogy(pos_f, periodogram, label='Periodograma')
plt.semilogy(freq, welch, label='Welch')
plt.title('PDS')
plt.title('Welch')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend()

f_mask = (f >= 0) & (f <= Bw)
trunc_spectrum = spectrum
trunc_spectrum[~f_mask] = 0
ecg_rebuilt = np.fft.ifft(trunc_spectrum, axis=0)

plt.figure()
plt.plot(np.real(ecg_rebuilt))
plt.grid(True)
plt.title('Señal temporal reconstruida')
plt.show()