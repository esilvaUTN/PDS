import numpy as np
import matplotlib.pyplot as plt
 
# Datos generales de la simulación
N = 1000    #Cantidad de muestras
fs = N
R = 200     #Cantidad de realizaciones

a1 = 2              #Amplitud de la señal original
SNR_db = [3, 10]    #Relación señal a ruido

for snr in SNR_db:
    SNR = 10**(snr/10)      #SNR en veces
    Pn = ((a1**2)/2)/SNR    #Potencia de ruido
    
    fr = np.random.uniform(-0.5, 0.5, R)
    w0 = N/4
    w1 = w0 + fr          #Frecuencia de señal
    
    k = np.arange(0, N/fs, 1/fs)        #Vector de tiempo
    
    s = np.zeros((N, R))
    n = np.zeros((N, R))
    
    for i in range(R):
        W = w1[i]
        s[:, i] = a1*np.sin(2*np.pi*W*k)                        #Generación de señal senoidal
        n[:, i] = np.random.normal(0, np.sqrt(Pn), N)     #Ruido aleatorio analógico
    
    x = s + n   #Señal con ruido
    
    ft_X = np.fft.fft(x, axis=0)/N      #Espectro
    f = np.fft.fftfreq(N, 1/fs)    #Vector de frecuencias
    
    plt.figure()
    plt.plot(f, 10*np.log10(np.abs(ft_X)))
    plt.title(f'Espectro de 200 realizaciones con SNR = {snr} dB')
    plt.xlabel('Módulo')
    plt.ylabel('f')
    
    a = 2*np.abs(ft_X[N//4, :])            #Estimador de módulo
    plt.figure(2)
    plt.hist(a, bins=20, alpha=0.8, label=f'{snr} dB')
    plt.title('Histograma del Estimador de Módulo')
    plt.xlabel('Módulo')
    plt.ylabel('N')
    plt.legend()
    plt.show()
    
    index = np.argmax(np.abs(ft_X[:fs//2, :]), axis=0)  #Estimador de fase
    w = f[index]
    plt.figure(3)
    plt.hist(index, bins=20, alpha=0.8, label=f'{snr} dB')
    plt.title('Histograma del Estimador de Fase')
    plt.xlabel('Fase')
    plt.ylabel('N')
    plt.legend()
    plt.show()
    