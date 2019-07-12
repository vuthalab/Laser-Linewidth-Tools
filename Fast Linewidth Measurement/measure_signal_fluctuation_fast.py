import sys
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import os
os.chdir("/home/labuser/Desktop/googledrive/code/Samarium_control/Widgets/Objects")
from digitizer import Digitizer                        # Keysight digitizer
os.chdir("/home/labuser/Desktop/googledrive/code/Samarium_control/Widgets/Experiment Functions")
from UsefulFunctions import *

signal_voltage_range = 0.5 # V
signal_channel = 1
T_measure = 10e-3 # s
sampling_rate = 20e6 #Maximum sampling rate on digitizer
n_trials = 16
num_samples = T_measure * sampling_rate

## Create Folder
run_type = 'Line Width'

file_path = nameNoiseFolder(run_type, 'Noise Measurement') #Gets name and saves folder
os.makedirs(file_path)
os.chdir(file_path)

## Load Digitizer
DG = Digitizer('/dev/digitizer', ch1_voltage_range= signal_voltage_range, ch2_voltage_range=4, num_samples=num_samples, sampling_rate=sampling_rate, data_format='float32', coupling = 'AC') # float32 returns voltages directly, no conversion required
DG.status_report()

print ("Digitizer clock status: " + str(DG.clock_status())) # Make sure that the digitizer is receiving is an external clock. Or change the digitizer.py file to have it run on an internal clock

## Acquire Data
for i in range(n_trials):
    print ('Trial ' + str(i))
    V1 = DG.get_single_channel_waveform(signal_channel)
    np.save(file_path + "Trial " + str(i) + ".npy", V1)