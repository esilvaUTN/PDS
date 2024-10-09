import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Datos generales de la simulación
N = 1000    #Cantidad de muestras
fs = N
R = 200     #Cantidad de realizaciones

a1 = 2      #Amplitud de la señal original

fr = np.random.uniform(-2, 2, R)
w0 = N/4
w1 = w0 + fr          #Frecuencia de señal

k = np.arange(0, N/fs, 1/fs)        #Vector de tiempo

x = np.zeros((N, R))
w_blackman = np.zeros((N, R))
w_flattop = np.zeros((N, R))

for i in range(R):
    W = w1[i]
    x[:, i] = a1*np.sin(2*np.pi*W*k)                        #Generación de señal senoidal
        
    w_blackman[:, i] = sp.signal.windows.blackmanharris(N)  #Ventana Blackman Harris
    w_flattop[:, i] = sp.signal.windows.flattop(N)          #Ventana Flat Top

f = np.arange(0, fs, 1)             #Vector de frecuencias
ft_X_rect = np.fft.fft(x, axis=0)/N      #Espectro
ft_X_blackman = np.fft.fft(x*w_blackman, axis=0)/N      #Espectro
ft_X_flattop = np.fft.fft(x*w_flattop, axis=0)/N      #Espectro

a_rect = 2*np.abs(ft_X_rect[N//4, :])        #Estimador de módulo
a_blackman = 2*np.abs(ft_X_blackman[N//4, :])        #Estimador de módulo
a_flattop = 2*np.abs(ft_X_flattop[N//4, :])        #Estimador de módulo

plt.figure()
plt.hist(a_rect, bins=20, alpha=0.8, label='Ventana Rectangular', color='blue')
plt.hist(a_blackman, bins=20, alpha=0.8, label='Ventana Blackman Harris', color='orange')
plt.hist(a_flattop, bins=20, alpha=0.8, label='Ventana Flat Top', color='green')
plt.title('Histograma del Estimador de Módulo')
plt.xlabel('Módulo')
plt.ylabel('N')
plt.legend()
plt.show()

index_rect = np.argmax(np.abs(ft_X_rect[:fs//2, :]), axis=0)  #Estimador de frecuencia
w_rect = f[index_rect]
index_blackman = np.argmax(np.abs(ft_X_blackman[:fs//2, :]), axis=0)  #Estimador de frecuencia
w_blackman = f[index_blackman]
index_flattop = np.argmax(np.abs(ft_X_flattop[:fs//2, :]), axis=0)  #Estimador de frecuencia
w_flattop = f[index_flattop]

plt.figure()
plt.hist(w_rect, bins=20, alpha=0.8, label='Ventana Rectangular', color='blue')
plt.hist(w_blackman, bins=20, alpha=0.8, label='Ventana Blackman Harris', color='orange')
plt.hist(w_flattop, bins=20, alpha=0.8, label='Ventana Flat Top', color='green')
plt.title('Histograma del Estimador de Frecuencia')
plt.xlabel('Frecuencia')
plt.ylabel('N')
plt.legend()
plt.show()

print("Sesgo ventana rectanfular: " + str(np.mean(a_rect) - a1))
print("Varianza ventana rectangular: " + str(np.var(a_rect)) + "\n")
print("Sesgo ventana Blackman Harris: " + str(np.mean(a_blackman) - a1))
print("Varianza ventana Blackman Harris: " + str(np.var(a_blackman)) + "\n")
print("Sesgo ventana Flat Top: " + str(np.mean(a_flattop) - a1))
print("Varianza ventana Flat Top: " + str(np.var(a_flattop)) + "\n")