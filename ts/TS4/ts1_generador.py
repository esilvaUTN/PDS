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
        xx = dc + vmax*np.sin(2*np.pi*ff*tt + ph)      #Generación de señal senoidal
    elif signal == 'square':
        xx = dc + vmax*sp.signal.square(2*np.pi*ff*tt + ph) #Generación de señal rectanular
    
    return tt, xx