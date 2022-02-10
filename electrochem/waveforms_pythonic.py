import sys
from ctypes import *
from dwfconstants import *
from munch import Munch
import time

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
    
    def tearDown(self):
        dwf.FDwfAnalogOutReset(self.hdwf, c_int(0))
        dwf.FDwfDeviceCloseAll()
        self.info['closed_device'] = True


    def setup_Network_Analyzer(self, steps: int, start: float,
                                stop: float, reference: float, amplitude: float, periods):
        dwf.FDwfDeviceAutoConfigureSet(self.hdwf, c_int(3)) # this option will enable dynamic adjustment of analog out settings like: frequency, amplitude...
        dwf.FDwfAnalogImpedanceReset(self.hdwf)
        # These two parameters don't matter for us I believe
        dwf.FDwfAnalogImpedanceModeSet(self.hdwf, c_int(0)) # 0 = W1-C1-DUT-C2-R-GND, 1 = W1-C1-R-C2-DUT-GND, 8 = AD IA adapter
        dwf.FDwfAnalogImpedanceReferenceSet(self.hdwf, c_double(reference)) # reference resistor value in Ohms

        dwf.FDwfAnalogImpedancePeriodSet(self.hdwf, c_int(8))

        dwf.FDwfAnalogImpedanceFrequencySet(self.hdwf, c_double(start)) # frequency in Hertz
        dwf.FDwfAnalogImpedanceAmplitudeSet(self.hdwf, c_double(amplitude)) # 1V amplitude = 2V peak2peak signal
        dwf.FDwfAnalogImpedanceConfigure(self.hdwf, c_int(1)) # start


        rgHz = np.geomspace(start, stop, steps)
        rgGaC1 = np.zeros(steps)
        rgGaC2 = np.zeros(steps)
        rgPhC2 = np.zeros(steps)

        sts = c_byte()
        szerr = create_string_buffer(512)

        # Should be in a thread...
        for i in range(steps):
            hz = rgHz[i]
            dwf.FDwfAnalogImpedanceFrequencySet(self.hdwf, c_double(hz)) # frequency in Hertz
            time.sleep(0.01)
            dwf.FDwfAnalogImpedanceStatus(self.hdwf, None) # ignore last capture since we changed the frequency
            # I need to think about how to increase the delay in cycles
            while True:
                if dwf.FDwfAnalogImpedanceStatus(self.hdwf, byref(sts)) == 0:
                    dwf.FDwfGetLastErrorMsg(szerr)
                    print(str(szerr.value))
                    quit()
                if sts.value == 2:
                    break
                time.sleep(0.01)
            
            gain1 = c_double()
            gain2 = c_double()
            phase2 = c_double()
            dwf.FDwfAnalogImpedanceStatusInput(self.hdwf, c_int(0), byref(gain1), 0) # relative to FDwfAnalogImpedanceAmplitudeSet Amplitude/C1
            dwf.FDwfAnalogImpedanceStatusInput(self.hdwf, c_int(1), byref(gain2), byref(phase2)) # relative to Channel 1, C1/C#

            rgGaC1[i] = 1.0/gain1.value
            rgGaC2[i] = 1.0/gain2.value
            rgPhC2[i] = -phase2.value # Leave it in volts...
            # peak voltage value:
            # rgGaC1[i] = amplitude/gain1.value 
            # rgGaC2[i] = amplitude/gain1.value/gain2.value 

        # Don't I have a real function for this?

        # Output is just the voltage that came in (potentially a factor of 10 bigger / smaller?)

        dwf.FDwfAnalogImpedanceConfigure(self, c_int(0)) # stop

        return rgHz, rgGaC1, rgGaC2, rgPhC2

if __name__ == '__main__':
    ad = AnalogDiscovery()
    ad.setup()
    ad.tearDown()
    print(ad.info)
