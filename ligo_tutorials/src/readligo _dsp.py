#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 09:24:42 2019

@author: chaitanya
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import readligo as rl

"""Loading Data""" 
filename = 'H-H1_LOSC_4_V1-815411200-4096.hdf5'
strain,time,channel_dict = rl.loaddata(filename)
dt = time[1] - time[0]
fs = int(1/dt)

"""Segmentation"""
segList = rl.dq_channel_to_seglist(channel_dict['DEFAULT'],fs)
length = 16                                                                     #Determines number of samples or data packets where each data packet has 4096 samples
strain_seg = strain[segList[0]][0:fs*length]
time_seg = time[segList[0]][0:fs*length]

"""Time Series"""
plt.figure()
plt.plot(time_seg - time_seg[0],strain_seg)
plt.xlabel('Time since GPS' + str(time_seg[0]))
plt.ylabel('Strain')
plt.show()

"""Fourier Transfor"""
window = np.blackman(strain_seg.size)
windowed_strain = strain_seg*window
frequency_domain = np.fft.rfft(windowed_strain)/fs
frequency = np.fft.rfftfreq(len(windowed_strain))*fs

plt.figure()
plt.grid('on')
plt.loglog(frequency,abs(frequency_domain))
plt.axis([10, fs/2.0, 1e-24, 1e-18])
plt.xlabel('frequency(Hz)')
plt.ylabel('Strain')
plt.show()

"""PSD and ASD"""
plt.figure()
plt.grid('on')
Pxx,freqs = mlab.psd(strain_seg,Fs=fs,NFFT=fs)    
plt.loglog(freqs,Pxx)
plt.axis([10, fs/2.0, 1e-46, 1e-36])
plt.xlabel('frequency(Hz)')
plt.ylabel('PSD')
plt.show()

plt.figure()
plt.grid('on')
plt.loglog(freqs,np.sqrt(Pxx))
plt.axis([10, fs/2.0, 1e-23, 1e-18])
plt.xlabel('frequency(Hz)')
plt.ylabel('ASD')
plt.show()

"""Spectrogram"""
NFFT = 1024
window = np.blackman(NFFT)
plt.figure()
spec_power,freq,bins,im = plt.specgram(strain_seg,NFFT=NFFT,Fs=fs,window=window)
plt.xlabel('Time(s)')
plt.ylabel('frequency(Hz)')
plt.show()