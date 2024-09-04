# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:45:37 2024

@author: elosi
"""

import numpy as np

from ts1_generador import signal_generator
import matplotlib.pyplot as plt    

fs = 800        #Frecuencia de muestro
N = fs          #Cantidad de muestras para k=1
f0 = 1          #Frecuencia de la señal
A = 1.5         #Amplitud de la señal original
DC = 0
phi = 0

t, Sr = signal_generator(A, DC, f0, phi, N, fs) #Llamada a la función generador

B = [4, 8, 16]      #Numero de bits
Vf = 2              #Tensión full scale

for b in B:
    q = 2*Vf/((2**b)-1) #Paso de cuantización
    Sq = np.round(Sr/q)*q
    
    plt.figure()
    plt.plot(t, Sr)
    plt.title('Señal original y cuantizada')
    plt.xlabel('t')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()
    
    plt.plot(t, Sq)
    plt.show()

b = 4                   #Numero de bits
q = 2*Vf/((2**b)-1)     #Paso de cuantización
Sq = np.round(Sr/q)*q

e = Sq - Sr
print("Media de la señal de error: ", np.mean(e))
print("Desvío de la señal de error: ", np.std(e))

q_uniform = np.random.uniform(-q/2, q/2, size=Sq.shape)
print("\nMedia de una distribución uniforme en el rango -q/2 ; q/2: ", np.mean(q_uniform))
print("Desvìo de una distribución uniforme en el rango -q/2 ; q/2: ", np.std(q_uniform))

plt.figure()
plt.plot(t, e)
plt.title('Señal de error para B = 4')
plt.xlabel('t')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()