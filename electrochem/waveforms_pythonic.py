from ctypes import *
from dwfconstants import *
from munch import Munch

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

def



def GetVersion():
    version = create_string_buffer(16)
    dwf.FDwfGetVersion(version)
    return str(version.value)



class AnalogDiscovery:
    def __init__(self):
        self.info = Munch()

    

    def setup(self):

    
