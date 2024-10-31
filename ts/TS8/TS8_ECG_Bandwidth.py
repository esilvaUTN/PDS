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

ecg_full = vertical_flaten(mat_struct['ecg_lead'])
N_full = len(ecg_full)
ecg_trim = ecg_full[0:10000]                    #Primeros 10 segundos
N_trim = len(ecg_trim)

qrs_detections = mat_struct['qrs_detections']
qrs_pattern = mat_struct['qrs_pattern1']

N_qrs = 600
max_hb = max(abs(ecg_full))
qrs_hb = np.zeros((N_qrs, len(qrs_detections)))
for i in range(len(qrs_detections)):
    hb = int(qrs_detections[i])
    qrs_hb[:, i] = (ecg_full[hb-250:hb+350].squeeze()-np.mean(ecg_full[hb-250:hb+350]))

plt.figure()
plt.plot(ecg_full)
plt.title('ECG Entero')
plt.figure()
plt.plot(ecg_trim)
plt.title('ECG primeros 10 segundos')
plt.figure()
plt.plot(qrs_hb)
plt.title('Realizaciones de latidos en una ventana')

hb_normal = mat_struct['heartbeat_pattern1']
hb_normal = hb_normal/max(abs(hb_normal))

hb_ventricular = mat_struct['heartbeat_pattern2']
hb_ventricular = hb_ventricular/max(abs(hb_ventricular))

plt.figure()
plt.plot(hb_normal, label='Latido Normal')
plt.plot(hb_ventricular, label='Latido Ventricular')
plt.title('Latidos por tipo')
plt.xlabel('t [Hz]')
plt.ylabel('Amplitud Normalizada')
plt.legend()

freq_full, welch_full = sp.signal.welch(ecg_full, fs=fs, nperseg=N_full/5, axis=0)
freq_trim, welch_trim = sp.signal.welch(ecg_trim, fs=fs, nperseg=N_trim/5, axis=0)
freq_qrs, welch_qrs = sp.signal.welch(qrs_hb, fs=fs, nperseg=N_qrs/5, axis=0)

welch_full = welch_full/np.sum(welch_full)
welch_trim = welch_trim/np.sum(welch_trim)
welch_qrs = welch_qrs/np.sum(welch_qrs)

plt.figure();
plt.semilogy(freq_full, welch_full, label='Welch Full ECG')
plt.semilogy(freq_trim, welch_trim, label='Welch 10s Trimmed ECG')
plt.title('PDS')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend()

qrs_mean = np.mean(welch_qrs, axis=1)

plt.figure();
plt.semilogy(freq_qrs, welch_qrs, color='blue')
plt.semilogy(freq_qrs, qrs_mean, color='orange', label='Promedio')
plt.title('PDS. Realizaciones de latidos')
plt.xlabel('f [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend()

qrs_mean = qrs_mean/np.sum(qrs_mean)
Paccum_welch = np.cumsum(qrs_mean)
thresh = 0.99
index = np.where(Paccum_welch >= thresh)[0][0]
Bw = freq_qrs[index]
print('Ancho de banda a partir de Welch: ' + str(Bw))
thresh = 0.8
index = np.where(Paccum_welch >= thresh)[0][0]
Bw = freq_qrs[index]
print('Ancho de banda a partir de Welch: ' + str(Bw))