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

signal_voltage_range = 0.25 # V
signal_channel = 1
T_measure = 10e-3 # s
sampling_rate = 20e6 #Maximum sampling rate on digitizer
n_trials = 16
num_samples = T_measure * sampling_rate

## Create Folder
run_type = 'Amplifier'

file_path = nameNoiseFolder(run_type, 'Noise Measurement') #Gets name and saves folder
os.makedirs(file_path)
os.chdir(file_path)

## Load Digitizer
DG = Digitizer('/dev/digitizer', ch1_voltage_range= signal_voltage_range, ch2_voltage_range=4, num_samples=num_samples, sampling_rate=sampling_rate, data_format='float32', coupling = 'AC') # float32 returns voltages directly, no conversion required
DG.status_report()

print ("Digitizer clock status: " + str(DG.clock_status())) # Make sure that the digitizer is receiving is an external clock. Or change the digitizer.py file to have it run on an internal clock

## Acquire Data
V1all = []
for i in range(n_trials):
    print ('Trial ' + str(i))
    V1 = DG.get_single_channel_waveform(signal_channel)
    V1all += [V1]
    np.save(file_path + "Trial " + str(i) + ".npy", V1)

## Define Analysis Functions / Parameters
from matplotlib import rc
import glob

def test_sine(t):
    return 2*np.sin(2*np.pi*1000*t) + 0.5*np.sin(2*np.pi*3500*t)

def lorentzian(x, b, a, x_0, sigma) :
    # A lorentzian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Full Width at Half Maximum   : p[3]
    return b+a/(1.0+((x-x_0)/sigma)**2)

N = sampling_rate * T_measure
dT = 1/sampling_rate
T = N*dT
df = 1.0/T

## FFT
S_nu_list = []
for signal in V1all:
    signal_ac = signal - np.average(signal)
    freq_noise = G_nu * signal_ac # Converting from voltage noise to frequency noise

    # FFT of signal
    yfft = np.fft.fftshift( np.fft.fft(freq_noise,norm="ortho") ) / np.sqrt(df/dT)  # watch out for this normalization factor
    f = np.fft.fftshift( np.fft.fftfreq(len(freq_noise), dT) )

    # # checking normalization using Parseval's theorem
    # t_sum = np.sum(np.abs(freq_noise)**2 * dT)
    # print "Time integral = ", t_sum
    # f_sum = np.sum(np.abs(yfft)**2 * df)
    # print "Frequency integral = ", f_sum
    # print f_sum/t_sum

    S_nu_trace = np.abs(yfft)**2 /T # Power spectal density of voltage
    S_nu_list.append(S_nu_trace)

S_nu = np.mean(S_nu_list, axis = 0)
W_nu = 2 * S_nu[f>0]
f_plus = f[f>0]

# check parseval again
# print np.sum(S_nu*df) / np.sum(W_nu*df)

## rms frequency, sanity check
delta_nu_rms = np.std(freq_noise)         # time domain rms
alt_nu_rms = np.sqrt( np.sum(W_nu * df) ) # freq domain rms

print (delta_nu_rms)
print (alt_nu_rms)

## frequency noise spectrum plot
fig, ax = plt.subplots()
ax.loglog(f_plus, np.sqrt(W_nu))
plt.xlabel('$f$ [Hz]')
plt.ylabel(r'$W_\nu^{1/2}$ [Hz/$\sqrt{\mathrm{Hz}}$]')
plt.grid()

for fmt in ['png','svg','pdf']:
    fig.savefig("frequency_noise." + fmt,format=fmt)

plt.show()