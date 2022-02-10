"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2019-07-12

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import time
import sys
import numpy
import numpy as np
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    dwf = cdll.LoadLibrary("dwf.dll")
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

hdwf = c_int()
szerr = create_string_buffer(512)
print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    print("failed to open device")
    quit()

# this option will enable dynamic adjustment of analog out settings like: frequency, amplitude...
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(3)) 

sts = c_byte()
# These are all parameters...
steps = 31
start = 10
stop = 10000
reference = 1e4
amplitude = 0.5
periods = 16 # Minimum periods to measure for

print("Frequency: "+str(start)+" Hz ... "+str(stop/1e3)+" kHz Steps: "+str(steps))
dwf.FDwfAnalogImpedanceReset(hdwf)
# These two parameters don't matter for us I believe
dwf.FDwfAnalogImpedanceModeSet(hdwf, c_int(0)) # 0 = W1-C1-DUT-C2-R-GND, 1 = W1-C1-R-C2-DUT-GND, 8 = AD IA adapter
dwf.FDwfAnalogImpedanceReferenceSet(hdwf, c_double(reference)) # reference resistor value in Ohms

dwf.FDwfAnalogImpedancePeriodSet(hdwf, c_int(8))

dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(start)) # frequency in Hertz
dwf.FDwfAnalogImpedanceAmplitudeSet(hdwf, c_double(amplitude)) # 1V amplitude = 2V peak2peak signal
dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(1)) # start

time.sleep(2)

rgHz = np.zeros(steps)
rgGaC1 = np.zeros(steps)
rgGaC2 = np.zeros(steps)
rgPhC2 = np.zeros(steps)

# Should be in a thread...
for i in range(steps):
    hz = stop * pow(10.0, 1.0*(1.0*i/(steps-1)-1)*math.log10(stop/start)) # exponential frequency steps
    rgHz[i] = hz
    dwf.FDwfAnalogImpedanceFrequencySet(hdwf, c_double(hz)) # frequency in Hertz
    time.sleep(0.01)
    dwf.FDwfAnalogImpedanceStatus(hdwf, None) # ignore last capture since we changed the frequency
    # I need to think about how to increase the delay in cycles
    while True:
        if dwf.FDwfAnalogImpedanceStatus(hdwf, byref(sts)) == 0:
            dwf.FDwfGetLastErrorMsg(szerr)
            print(str(szerr.value))
            quit()
        if sts.value == 2:
            break
    
    gain1 = c_double()
    gain2 = c_double()
    phase2 = c_double()
    dwf.FDwfAnalogImpedanceStatusInput(hdwf, c_int(0), byref(gain1), 0) # relative to FDwfAnalogImpedanceAmplitudeSet Amplitude/C1
    dwf.FDwfAnalogImpedanceStatusInput(hdwf, c_int(1), byref(gain2), byref(phase2)) # relative to Channel 1, C1/C#

    rgGaC1[i] = 1.0/gain1.value
    rgGaC2[i] = 1.0/gain2.value
    rgPhC2[i] = -phase2.value # Leave it in volts...
    # peak voltage value:
    # rgGaC1[i] = amplitude/gain1.value 
    # rgGaC2[i] = amplitude/gain1.value/gain2.value 

# Don't I have a real function for this?

# Output is just the voltage that came in (potentially a factor of 10 bigger / smaller?)

dwf.FDwfAnalogImpedanceConfigure(hdwf, c_int(0)) # stop
dwf.FDwfDeviceClose(hdwf)

plt.subplot(211)

plt.plot(rgHz, rgGaC1, color='orange')
plt.plot(rgHz, rgGaC2, color='blue')
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')

plt.subplot(212)
plt.plot(rgHz, rgPhC2)
ax = plt.gca()
ax.set_xscale('log')
plt.show()

