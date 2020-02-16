#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:02:05 2019

@author: chaitanya
"""
import gwpy
from gwpy.timeseries import TimeSeries
import gwosc
from gwosc import datasets as ds
import numpy as np
import matplotlib.pyplot as plt

gps = ds.event_gps('GW150914')
segment = (int(gps)-5,int(gps)+5)

data = TimeSeries.fetch_open_data('H1',*segment,verbose=True,tag = 'CLN')
print(data)

data.plot()

fft = data.fft()
print(fft)

plot = fft.abs().plot()
plot.show()

segment_run = ds.run_segment('S6')
print(segment_run)

months = gwpy.time.tconvert(segment_run[1]) - gwpy.time.tconvert(segment_run[0]) 