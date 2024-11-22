# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:08:17 2024

@author: elosi

Filtro con firls o con remes
Filtros no lineales mediana y cubicspline en zonas más afectadas

"""

# Inicialización e importación de módulos

# Módulos para Jupyter
import warnings
warnings.filterwarnings('ignore')

# Módulos importantantes
import scipy.signal as sig
import scipy.interpolate as interpol
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

fig_sz_x = 10
fig_sz_y = 7
fig_dpi = 100 # dpi

fig_font_size = 16

mpl.rcParams['figure.figsize'] = (fig_sz_x,fig_sz_y)
plt.rcParams.update({'font.size':fig_font_size})

###
## Señal de ECG registrada a 1 kHz, con contaminación de diversos orígenes.
###

# para listar las variables que hay en el archivo
#io.whosmat('ecg.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten()
cant_muestras = len(ecg_one_lead)

fs = 1000 # Hz
nyq_frec = fs / 2

# Plantilla

# filter design
ripple = 1 # dB
atenuacion = 20 # dB

ws1 = 0.1 #Hz
wp1 = 0.5 #Hz
wp2 = 30.0 #Hz
ws2 = 45.0 #Hz

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains/20)

#IIR
bp_sos_butter = sig.iirdesign([wp1, wp2], [ws1, ws2], ripple, atenuacion, ftype='butter', output='sos', fs=fs)

w, h = sig.sosfreqz(bp_sos_butter, worN=20000)
plt.plot((w/np.pi)*nyq_frec, 20*np.log10(np.abs(h)), label="Filtro IIR")
plt.title('Respuesta en Frecuencia')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Modulo [dB]')
plt.grid()
    
ECG_f_butt = sig.sosfiltfilt(bp_sos_butter, ecg_one_lead)

demora = 0

# Segmentos de interés con ALTA contaminación
regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    med_filt = sig.medfilt(sig.medfilt(ecg_one_lead[zoom_region], 201), 601)
    ecg_filt = ecg_one_lead[zoom_region] - med_filt
    
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ECG_f_butt[zoom_region + demora], label='Win')
    plt.plot(zoom_region, ecg_filt, label='Median')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()

#FIR
cant_coeficientes = 10001
num_win = sig.firwin2(cant_coeficientes, frecs, gains, window='blackmanharris')
den = 1.0

w, h = sig.freqz(num_win, worN=20000)
#plt.figure()
#plt.plot((w/np.pi)*nyq_frec, 20*np.log10(np.abs(h)), label="Filtro FIR")
#plt.legend()

#ECG_f_win = sig.filtfilt(num_win, den, ecg_one_lead)

qrs_detections = mat_struct['qrs_detections']
k = np.zeros(len(qrs_detections))
v = np.zeros(len(qrs_detections))
for i in range(len(qrs_detections)):
    hb = int(qrs_detections[i])-100
    k[i] = hb
    v[i] = np.mean(ecg_one_lead[hb-10:hb+10])

# Segmentos de interés con ALTA contaminación
regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    ecg_baseline = interpol.CubicSpline(k, v)
    t_interpol = np.linspace(zoom_region.min(), zoom_region.max(), 10*len(zoom_region))
    
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG')
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='Win')
    plt.plot(t_interpol, ecg_baseline(t_interpol), label="Baseline Interpolation")
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()