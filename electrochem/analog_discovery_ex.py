"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2018-07-19

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
import time

from dwfconstants import *
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import plotly.express as px

import threading

import PySimpleGUI as sg

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")


np = numpy

def make_filename(values, key):     
    today = datetime.today()
    datestring = today.strftime("%Y-%m-%d %H-%M")
    directory, basefilename = os.path.split(values[key])
    return os.path.join(directory, f"{datestring} {basefilename}")


def setupAnalogDiscovery():

    info = dict()
    version = create_string_buffer(16)
    dwf.FDwfGetVersion(version)
    info.version = str(version.value)

    cdevices = c_int()
    dwf.FDwfEnum(c_int(0), byref(cdevices))
    info['number_of_devices'] = str(cdevices.value)

    if cdevices.value > 0:
        print("Opening first device")
        # 
        hdwf = c_int()
        dwf.FDwfDeviceOpen(c_int(0), byref(hdwf))

        if hdwf.value == hdwfNone.value:
            print("failed to open device")
            info['opened_device'] = False
        else:
            info['opened_device'] = True
    else:
        hdwf = None # No device...
    
    return hdwf, info

def setupOutputsDC(hdwf, V_ch1, V_ch2):

    # this option will enable dynamic adjustment of analog out settings like: frequency, amplitude...
    dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(3)) 

    funcDC = c_int(0)

    print("Configure and start first analog out channel")
    dwf.FDwfAnalogOutEnableSet(hdwf, c_int(0), c_int(1)) 
    dwf.FDwfAnalogOutEnableSet(hdwf, c_int(1), c_int(1)) 
    dwf.FDwfAnalogOutFunctionSet(hdwf, c_int(0), funcDC) # 1 = Sine wave
    dwf.FDwfAnalogOutFunctionSet(hdwf, c_int(1), funcDC) # 1 = Sine wave

    dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(0), c_double(V_ch1))
    dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(1), c_double(V_ch2))

    dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(1))
    dwf.FDwfAnalogOutConfigure(hdwf, c_int(1), c_int(1))


def setupInputs(hdwf, frequency=1e6, buffer=1000, range=5):
    print("Configure analog in")
    dwf.FDwfAnalogInFrequencySet(hdwf, c_double(frequency))
    print("Set range for all channels")
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(-1), c_double(range))
    dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(buffer))


def readData(hdwf, buffer=1000):
    # This needs to be a function
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))

    sts = c_int()
    while True:
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
        if sts.value == DwfStateDone.value :
            break
        time.sleep(0.1)

    rg = (c_double*buffer)()
    dwf.FDwfAnalogInStatusData(hdwf, c_int(0), rg, len(rg)) # get channel 1 data
    return rg


def cleanupAnalogDiscovery(hdwf):
    dwf.FDwfAnalogOutReset(hdwf, c_int(0))
    dwf.FDwfDeviceCloseAll()


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

# Full data now:
Vgate = np.linspace(-1,1,11)
Vdrain = np.linspace(-1,1,11)

def runAD(Vgate, Vdrain, filename, dataArray, delay_gate=0.1, delay_drain=0.1):

    hdwf, info = setupAnalogDiscovery()

    pnSizeMin, pnSizeMax = c_int(), c_int()
    dwf.FDwfAnalogInBufferSizeInfo(hdwf, byref(pnSizeMin), byref(pnSizeMax))

    print(f"min and max buffers: {pnSizeMin} {pnSizeMax}")

    pfsfilter = c_int()
    dwf.FDwfAnalogInChannelFilterGet(hdwf, c_int(0), byref(pfsfilter))

    print(f"Channel 1 Filter Info: {pfsfilter}")

    rgVoltsStep = (c_double*32)()
    pnSteps = c_int()
    dwf.FDwfAnalogInChannelRangeSteps(hdwf, byref(rgVoltsStep), byref(pnSteps))

    rgVoltsStep = np.array(list(rgVoltsStep))
    print(f"rgVoltsStep: {rgVoltsStep}")


    setupOutputsDC(hdwf, Vdrain[0], Vgate[0])
    setupInputs(hdwf, frequency=1e5, range=5.0)

    pvoltsRange = c_double()
    dwf.FDwfAnalogInChannelRangeGet(hdwf, c_int(0), byref(pvoltsRange))

    print(f"Channel 1 Range: {pvoltsRange}")

    time.sleep(2)

    for Vg in Vgate:

        dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(0), c_double(Vg))
        time.sleep(delay_gate)
        
        for Vd in Vdrain:
            dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(1), c_double(Vd))
            time.sleep(delay_drain)

            rg = readData(hdwf)

            dc = sum(rg)/len(rg)
            print("DC: "+str(dc)+"V")
            dataArray.append(dict(Vg=Vg, Vd=Vd, Isd_mA=dc))


    cleanupAnalogDiscovery(hdwf)





if __name__ == "__main__":
    font = "Helvetica 18"
    sg.set_options(font=font)
    text_size = (10, 1)
    param_size = (8, 1)
    def Text(*args, **kwargs):
        return sg.Text(*args, **kwargs, size=text_size)

    def Input(*args, **kwargs):
        return sg.Input(*args, **kwargs, size=param_size)

    def voltages(name, Vi=0, Vf=0.5, Npts=6):
        return [Text("Vinitial"), Input(key='-Vinitial-'+name, default_text=f"{Vi:.2f}" ),
                Text("Vfinal"), Input(key='-Vfinal-'+name, default_text=f"{Vf:.2f}"),
                Text('Npts'), Input(key='-Npts-'+name, default_text=f"{Npts}")]
    
    def getVoltageInfo(values, name):
        Vi = float(values['-Vinitial-'+name])
        Vf = float(values['-Vfinal-'+name])
        Npts = int(values['-Npts-'+name])
        return np.linspace(Vi, Vf, Npts)
    
    layout = [[sg.Text("Gate Voltage Vg", size=(3*(text_size[0]+param_size[0]),1), 
                    justification='left')],
                voltages("Vgate"),
                [sg.Text("Drain Voltage Vd", size=(3*(text_size[0]+param_size[0]),1), 
                    justification='left')],
                voltages("Vdrain"),
                [Text("Gate delay (s):"), Input(default_text=f"0.5", key='-gate-delay-'),
                 Text("Drain delay (s):"), Input(default_text=f"0.5", key='-drain-delay-')],
                [sg.Button("Run", size=(10,2)), sg.FileSaveAs("Save", target='-FILENAME-', default_extension=""),
                sg.Input(visible=False, enable_events=True, key='-FILENAME-'), ],
                [Text("Data points: ", key='-npts-')],
                [sg.Button('Exit', size=(10, 2), pad=((280, 0), 3))]]
    window = sg.Window("Window Title", layout, finalize=True)


    class Thread:
        def is_alive(self):
            return False
    thread = Thread()
    was_running = False
    while True:
        event, values = window.read(timeout=100)
        running = thread.is_alive()
        # Handle all of the possible events:
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        
        if event == 'Run' and not running:
            data = []
            running = True
            Vgate = getVoltageInfo(values, "Vgate")
            Vdrain = getVoltageInfo(values, "Vdrain")
            drain_delay = float(values['-drain-delay-'])
            gate_delay = float(values['-gate-delay-'])
            thread = threading.Thread(target=runAD, args=(Vgate, Vdrain, "f1", data, gate_delay, drain_delay), daemon=True)
            thread.start()
        
        if running:
            pts = len(data)
            window['-npts-'].update(f"Data points: {pts}")
        
        if was_running and not running:
            df = pd.DataFrame(data)
            df['Vg_str'] = [f"{x:.3f}" for x in df['Vg']]
            df['Vd_str'] = [f"{x:.3f}" for x in df['Vd']]

            fig = px.scatter(df, x='Vg', y='Isd_mA', color='Vd_str')
            fig.show()
        
        if event == '-FILENAME-':
            if not values['-FILENAME-']:
                sg.popup("File not saved.")
            else:
                df = pd.DataFrame(data)
                basename = make_filename(values, '-FILENAME-')
                output_filename = f'{basename}.xlsx'
                df.to_excel(output_filename)
                sg.popup(f"File saved to {output_filename}")
        
        was_running = running

        
    # main()