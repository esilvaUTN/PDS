# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:21:19 2024

@author: elosi
"""

import numpy as np
import scipy as sp

def signal_generator(vmax=1, dc=0, ff=1, ph=0, nn=1, fs=1, signal='sine'):
    tt = np.arange(0, nn/fs, 1/fs)  #Generación del vector de tiempo
    
    if signal == 'sine':
        xx = dc + vmax*np.sin(2*np.pi*ff*tt + phi)      #Generación de señal senoidal
    elif signal == 'square':
        xx = dc + vmax*sp.signal.square(2*np.pi*ff*tt + phi) #Generación de señal rectanular
    
    return tt, xx

import matplotlib.pyplot as plt

fs = 1000       #Frecuencia de muestro
A = 1           #Amplitud de la señal
DC = 0          #Nivel de DC de la señal
k = 10          #Resolución espectral
N = 1000        #Cantidad de muestras
f0 = k*fs/N     #Frecuencia de la señal
phi = np.pi/2   #Desfasaje en radianes
sig = 'sine'    #Tipo de señal

t, x = signal_generator(A, DC, f0, phi, N, fs, sig) #Llamada a la función generador

plt.plot(t, x)
plt.title('Señal senoidal de ' + str(f0) + 'Hz')
plt.xlabel('t [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()

A = 2
DC = 1
k = 20
f0 = k*fs/N
phi = 0
sig = 'square'

t, x = signal_generator(A, DC, f0, phi, N, fs, sig)

plt.plot(t, x)
plt.title('Señal cuadrada de ' + str(f0) + 'Hz')
plt.xlabel('t [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()