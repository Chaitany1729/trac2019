#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:55:31 2019
Dynamic Spectrum
@author: chaitanya
"""

import numpy as np
import matplotlib.pyplot as plt

x,y=np.loadtxt('vela_Pulsar.mbr',delimiter = ' ',unpack = True,dtype = float)

plt.figure()
plt.hist(x,bins=100)                                                           #Verfifying gaussian distrubution
plt.xlabel('Voltage Range')
plt.ylabel('Number of Occurence')
plt.title('Probability Distribution of Voltage - Gaussian')

freq_channels = 256 
time = 1000                                                                    #in millisecond
sampling_freq = 33 * 10 ** 6
integration_time = 1
packets = 60

ROWS = 2*freq_channels                                                         #Unpacking the data                                                       
COLS = int(len(y)/(ROWS))                                           
unpacked_data_y = np.reshape(y,(ROWS,COLS),order='F')
unpacked_data_x = np.reshape(x,(ROWS,COLS),order='F')
                 

fft_data_y = np.zeros([ROWS,COLS])
for i in range(COLS):
    fft_data_y[:,i] = np.fft.fft(unpacked_data_y[:,i])
fft_data_y = fft_data_y[0:freq_channels,:]                                     #Retaining only positive frequency channels

fft_data_x = np.zeros([ROWS,COLS])
for i in range(COLS):
    fft_data_x[:,i] = np.fft.fft(unpacked_data_x[:,i])
fft_data_x = fft_data_x[0:freq_channels,:]                                     #Retaining only positive frequency channels

mod = np.abs(fft_data_y)                                                       #Performing FFT column wise
mod_squared = np.multiply(mod,mod)                                             #Taking modulus and squaring

plt.figure()
plt.hist(np.reshape(mod_squared,(np.size(mod_squared),1)),bins = 100)          #Verifying exponential distribution
plt.xlabel('Values')
plt.ylabel('Number of Occurence')
plt.title('Probability Distribution of Intensity - Exponential')


dynamic_spectra = np.zeros([freq_channels,time])

for j in range(time):                                                          #Averaging across time
    for k in range(packets*j,(packets*(j+1))):
        dynamic_spectra[:,j] = dynamic_spectra[:,j] + mod_squared[:,k]

dynamic_spectra = dynamic_spectra / packets  

plt.matshow(dynamic_spectra,cmap = 'twilight',interpolation = "nearest")
plt.xlabel('Time (ms)')
plt.ylabel('Frequency Channels')
plt.ylim(0,freq_channels)
plt.xlim(0,time)
plt.title('Dynamic Spectrum')
plt.show()
plt.colorbar()

efficiency = np.zeros([freq_channels])

RFI_affected_channel = []
for channel in range(freq_channels):
    efficiency[channel] = np.divide(np.mean(dynamic_spectra[channel,:]),(np.std(dynamic_spectra[channel,:])*np.sqrt(packets)))
    


plt.figure()
plt.plot(range(1,freq_channels),efficiency[1:])
plt.axis([0,256,0,1])
plt.xlabel('Frequency Channels')
plt.ylabel('Efficiency')
plt.title('Efficiency Plot')


threshold = np.median(efficiency) - (0.95/np.sqrt(1000))

removed_channels_efficiency = efficiency
for channel in range(freq_channels):
    if efficiency[channel] <= threshold:
        removed_channels_efficiency[channel] = None
        RFI_affected_channel.append(channel)
        
plt.figure()
plt.plot(range(1,freq_channels),removed_channels_efficiency[1:])
plt.axis([0,256,0,1])
plt.xlabel('RFI Removed Frequency Channels')
plt.ylabel('Efficiency')
plt.title('Efficiency Plot RFI Removed Frequency Channels')


RFI_removed_dynamic_spectra = np.zeros([np.shape(dynamic_spectra)[0]-len(RFI_affected_channel),1000])
i = 0
for channel in range(freq_channels):
    if channel not in RFI_affected_channel:
        RFI_removed_dynamic_spectra[i] = dynamic_spectra[channel]
        i = i + 1
        
plt.matshow(RFI_removed_dynamic_spectra,cmap = 'hot',interpolation = "nearest")
plt.xlabel('Time (ms)')
plt.ylabel('Frequency Channels')
plt.ylim(0,freq_channels - len(RFI_affected_channel))
plt.xlim(0,time)
plt.title('Dynamic Spectrum RFI Removed Frequency Channels')
plt.show()
plt.colorbar()

cross_cor = np.multiply(fft_data_x,np.conjugate(fft_data_y))

cross_dynamic_spectra = np.zeros([freq_channels,time])
for j in range(time):                                                          #Averaging across time
    for k in range(packets*j,(packets*(j+1))):
        cross_dynamic_spectra[:,j] = cross_dynamic_spectra[:,j] + cross_cor[:,k]

cross_dynamic_spectra = cross_dynamic_spectra / packets  

plt.matshow(cross_dynamic_spectra,cmap = 'summer',interpolation = "nearest")
plt.xlabel('Time (ms)')
plt.ylabel('Frequency Channels')
plt.ylim(0,freq_channels)
plt.xlim(0,time)
plt.title('Cross Correlated Dynamic Spectrum')
plt.show()
plt.colorbar()