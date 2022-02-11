import sys
from ctypes import *
from dwfconstants import *
from munch import Munch
import time

from dataclasses import dataclass

import numpy as np

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")


def Enum(device : int = 0):
    cdevices = c_int()
    dwf.FDwfEnum(c_int(device), byref(cdevices))
    return int(cdevices.value)

def GetVersion():
    version = create_string_buffer(16)
    dwf.FDwfGetVersion(version)
    return str(version.value)

def getImpedance(hdwf, channel: int):
    gain = c_double()
    phase = c_double()
    dwf.FDwfAnalogImpedanceStatusInput(hdwf, c_int(channel), byref(gain), byref(phase)) # relative to FDwfAnalogImpedanceAmplitudeSet Amplitude/C1
    return gain.value, phase.value

@dataclass
class Network:
    f: np.ndarray
    ch1: np.ndarray
    ch2: np.ndarray
    phase: np.ndarray


class AnalogDiscovery:
    def __init__(self):
        self.info = Munch()
        self.info.version = GetVersion()
        self.hdwf = c_int(hdwfNone.value)

    def open(self, device=0):
        dwf.FDwfDeviceOpen(c_int(device), byref(self.hdwf))
        if self.hdwf.value == hdwfNone.value:
            print("failed to open device")
            self.info['opened_device'] = False
        else:
            self.info['opened_device'] = True
        return self.hdwf.value

    def setup(self, device=0):
        self.info.n_devices = Enum()
        if self.info.n_devices > 0:
            self.open(device)
    
    def AnalogImpedanceStatusInput(self, channel: int):
        return getImpedance(self.hdwf, channel)
    
    def tearDown(self):
        dwf.FDwfAnalogOutReset(self.hdwf, c_int(0))
        dwf.FDwfDeviceCloseAll()
        self.info['closed_device'] = True

    def AnalogImpedanceStatus(self):
        sts = c_byte()
        szerr = create_string_buffer(512)
        status = dwf.FDwfAnalogImpedanceStatus(self.hdwf, byref(sts))
        if status == 0:
            dwf.FDwfGetLastErrorMsg(szerr)
        return sts.value, str(szerr.value)
    
    def AnalogImpedanceFrequencySet(self, f_Hz: float):
        dwf.FDwfAnalogImpedanceFrequencySet(self.hdwf, c_double(f_Hz)) # frequency in Hertz
    
    def AnalogImpedancePeriodSet(self, cycles: int):
        dwf.FDwfAnalogImpedancePeriodSet(self.hdwf, c_int(cycles))
    
    def AnalogImpedanceFrequencySet(self, start: float):
        dwf.FDwfAnalogImpedanceFrequencySet(self.hdwf, c_double(start)) # frequency in Hertz

    def AnalogImpedanceAmplitudeSet(self, amplitude: float):
        """Amplitude is the zero-to-peak amplitude."""
        dwf.FDwfAnalogImpedanceAmplitudeSet(self.hdwf, c_double(amplitude)) 
    
    def AnalogImpedanceReset(self):
        dwf.FDwfAnalogImpedanceReset(self.hdwf)

    def DeviceAutoConfigureSet(self, setting=3):
        dwf.FDwfDeviceAutoConfigureSet(self.hdwf, c_int(setting)) # this option will enable dynamic adjustment of analog out settings like: frequency, amplitude...


    def AnalogImpedanceConfigure(self, on=True):
        val = 1 if on else 0
        dwf.FDwfAnalogImpedanceConfigure(self.hdwf, c_int(val)) # start

    def AnalogOutNodeEnableSet(self, channel: int, node: int, enable: bool)
        dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_int(channel), c_int(node), c_bool(enable))

    def FDwfAnalogOutNodeFunctionSet(self, channel: int, node: int, func: int)
        dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, c_int(channel), c_int(node), c_int(func))

    def setup_Network_Analyzer(self, steps: int, start: float,
                                stop: float, amplitude: float, periods: int):

        self.DeviceAutoConfigureSet(3)
        self.AnalogImpedanceReset()
        self.AnalogImpedancePeriodSet(periods)
        self.AnalogImpedanceFrequencySet(start) # frequency in Hertz
        self.AnalogImpedanceAmplitudeSet(amplitude) # Zero to peak

        self.AnalogImpedanceConfigure(on=True) # start


        freq = np.geomspace(start, stop, steps)
        gainC1 = np.zeros(steps)
        gainC2 = np.zeros(steps)
        phaseC2 = np.zeros(steps)

        sts = c_byte()
        szerr = create_string_buffer(512)

        # Should be in a thread...
        for i in range(steps):
            hz = freq[i]
            self.AnalogImpedanceFrequencySet(hz)
            time.sleep(0.01)
            self.AnalogImpedanceStatusInput(0)  # I need to think about how to increase the delay in cycles
            while True:
                if self.AnalogImpedanceStatus()[0] == 2:
                    break
                time.sleep(0.01)

            g1, _ = self.AnalogImpedanceStatusInput(0)
            g2, p2 = self.AnalogImpedanceStatusInput(1)



            gainC1[i] = 1.0/g1
            gainC2[i] = 1.0/g2 # Relative to channel 1....
            phaseC2[i] = -p2 # Leave it in volts...
            
            # peak voltage value:
            # GaC1[i] = amplitude/gain1.value 
            # GaC2[i] = amplitude/gain1.value/gain2.value 

        # Don't I have a real function for this?

        # Output is just the voltage that came in (potentially a factor of 10 bigger / smaller?)

        self.AnalogImpedanceConfigure(on=False)

        return Network(f=freq, ch1=gainC1, ch2=gainC2, phase=phaseC2)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ad = AnalogDiscovery()
    ad.setup()
    values = ad.setup_Network_Analyzer(51, 100, 100000, 1, 16)
    print(values.f)
    print(values.ch1)
    print(values.ch2)
    print(values.phase)

    ad.tearDown()
    print(ad.info)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.loglog(values.f, values.ch1)
    ax1.loglog(values.f, values.ch2)
    ax2.semilogx(values.f, values.phase*180/np.pi)
    plt.show()
