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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy
import pandas as pd
import plotly.express as px
from munch import Munch

import threading
import signal

exit_event = threading.Event()
stop_event = threading.Event()
change_voltage_event = threading.Event()

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

def wvEnum():
    cdevices = c_int()
    dwf.FDwfEnum(c_int(0), byref(cdevices))
    return int(cdevices.value)

def signal_handler(signum, frame):
    exit_event.set()

signal.signal(signal.SIGINT, signal_handler)

def setupAnalogDiscovery():

    info = Munch()
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
    # master enable
    dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))
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
    return np.array(rg)


def cleanupAnalogDiscovery(hdwf):
    dwf.FDwfAnalogOutReset(hdwf, c_int(0))
    dwf.FDwfDeviceCloseAll()
    print("\nClosed Analog Discovery")


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

# Full data now:
Vgate = np.linspace(-1,1,11)
Vdrain = np.linspace(-1,1,11)

bufferSize = 1000
frequency = 1e5

# Rather than run directly, we should grab the values from the front end...

# Experiments layer...
#   - Does the experiment
#   - Saves data to someplace

# - Start the experiment (configure settings, etc)
# - Continue running the experiment... (each "trial") once everything is set up
    # - Update applied voltages if needed...
    # - 

class Experiments:
    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = axes
        self.hdwf = None
        self.info = []

    def setUp(self):
        self.hdwf, info = setupAnalogDiscovery()
        self.info.append(info)
        if self.hdwf is not None:
            return True
        else:
            return False

    def transferCurves(self, Vgate, Vdrain, Rfeedback, filename, dataArray,
            delay_gate=0.1, delay_drain=0.1, currentTraces=[None,None]):
        
        self.x_col = 'Vd'
        self.y_col = 'Isd_mA'
        
        # Set up...
        if self.hdwf is None:
            self.setUp()
        
        if self.hdwf is None:
            print("No device:")
            self.info.append('transferCurves failed - No device')
            return 0
        
        hdwf = self.hdwf

        # Set up experiment:
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
        setupInputs(hdwf, frequency=frequency, range=5.0,  buffer=bufferSize)

        pvoltsRange = c_double()
        dwf.FDwfAnalogInChannelRangeGet(hdwf, c_int(0), byref(pvoltsRange))

        print(f"Channel 1 Range: {pvoltsRange}")

        if ((ret_val := self.handleEvents()) < 0):
            return ret_val

        time.sleep(2)

        self.axes[1].clear()
        self.axes[1].set_xlabel("Vdrain (V)")
        self.axes[1].set_ylabel("$I_\\mathrm{sd}$ (mA)")
        
        start = time.time_ns()

        ## Enter the loop!
        for Vg in Vgate:

            dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(0), c_double(Vg))
            time.sleep(delay_gate)        
            
            for Vd in Vdrain:
                
                if ((ret_val := self.handleEvents()) < 0):
                    return ret_val

                dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(1), c_double(Vd))
                time.sleep(delay_drain)

                rg = readData(hdwf)

                currentTraces[0] = currentTraces[1]
                currentTraces[1] = rg
                

                dc = sum(rg)/len(rg)
                print("DC: "+str(dc)+"V")
                dataArray.append(dict(Vg=Vg, Vd=Vd, Isd_mA=dc/Rfeedback,
                                      t_s=(time.time_ns() - start)/1e9))
        
        self.stop()
        
    def data_logger(self, Vg, Vd, Rfeedback, frequency, buffer, dataArray,
                    currentTraces=[None,None], window=None):
        
        self.x_col = 't_s'
        self.y_col = 'Isd_mA'
        
        # Set up...
        if self.hdwf is None:
            self.setUp()


        if self.hdwf is None:
            print("No device:")
            self.info.append('transferCurves failed - No device')
            return 0
        
        hdwf = self.hdwf

        # Set up experiment:
        pnSizeMin, pnSizeMax = c_int(), c_int()
        dwf.FDwfAnalogInBufferSizeInfo(hdwf, byref(pnSizeMin), byref(pnSizeMax))

        print(f"min and max buffers: {pnSizeMin} {pnSizeMax}")

        pfsfilter = c_int()
        dwf.FDwfAnalogInChannelFilterGet(hdwf, c_int(0), byref(pfsfilter))

        print(f"Channel 1 Filter Info: {pfsfilter}")

        rgVoltsStep = (c_double*32)()
        pnSteps = c_int()
        dwf.FDwfAnalogInChannelRangeSteps(hdwf, byref(rgVoltsStep), byref(pnSteps))

        setupOutputsDC(hdwf, Vd, Vg)
        setupInputs(hdwf, frequency=frequency, range=5.0,  buffer=buffer)

        pvoltsRange = c_double()
        dwf.FDwfAnalogInChannelRangeGet(hdwf, c_int(0), byref(pvoltsRange))

        print(f"Channel 1 Range: {pvoltsRange}")

        if ((ret_val := self.handleEvents()) < 0):
            return ret_val

        dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(0), c_double(Vg))
        dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(1), c_double(Vd))


        time.sleep(2)

        self.axes[1].clear()
        self.axes[1].set_xlabel("Time (s) (V)")
        self.axes[1].set_ylabel("$I_\\mathrm{sd}$ (mA)")
        
        start = time.time_ns()

        while True:

            rg = readData(hdwf, buffer=buffer)

            currentTraces[0] = currentTraces[1]
            currentTraces[1] = rg

            dc = sum(rg)/len(rg)
            print("DC: "+str(dc)+"V")
            dataArray.append(dict(Vg=Vg, Vd=Vd, Isd_mA=dc/Rfeedback, t_s=(time.time_ns() - start)/1e9))
            
            if change_voltage_event.is_set():
                change_voltage_event.clear()
                Vg = float(window['-data_logger-Vg-'].get())
                Vd = float(window['-data_logger-Vd-'].get())
                dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(0), c_double(Vg))
                dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(1), c_double(Vd))

            if ((ret_val := self.handleEvents()) < 0):
                return ret_val
        
        self.stop()


    def handleEvents(self):
        ret_val = 0
        if exit_event.is_set():
            self.tearDown()
            exit_event.clear()
            ret_val += -1
        if stop_event.is_set():
            self.stop()
            stop_event.clear()
            ret_val += -2
        return ret_val

        
        

    def tearDown(self):
        if self.hdwf is not None:
            cleanupAnalogDiscovery(self.hdwf)
            self.hdwf = None
    
    def stop(self):
        if self.hdwf is not None:
            dwf.FDwfAnalogOutEnableSet(self.hdwf, c_int(-1), c_int(0)) # Disable all outputs...


def runAD(Vgate, Vdrain, Rfeedback, filename, dataArray, delay_gate=0.1, delay_drain=0.1, currentTraces=[None,None]):

    hdwf, info = setupAnalogDiscovery()
    print(hdwf)

    if hdwf is None:
        print("No device:") # Set in window?
        return 0

    if exit_event.is_set():
        cleanupAnalogDiscovery(hdwf)
        exit_event.clear()
        return 0

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
    setupInputs(hdwf, frequency=frequency, range=5.0,  buffer=bufferSize)

    pvoltsRange = c_double()
    dwf.FDwfAnalogInChannelRangeGet(hdwf, c_int(0), byref(pvoltsRange))

    print(f"Channel 1 Range: {pvoltsRange}")

    time.sleep(2)

    if exit_event.is_set():
        cleanupAnalogDiscovery(hdwf)
        exit_event.clear()
        return 0

    for Vg in Vgate:

        dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(0), c_double(Vg))
        time.sleep(delay_gate)        
        
        for Vd in Vdrain:
            
            if exit_event.is_set():
                cleanupAnalogDiscovery(hdwf)
                return 0

            dwf.FDwfAnalogOutOffsetSet(hdwf, c_int(1), c_double(Vd))
            time.sleep(delay_drain)

            rg = readData(hdwf)

            currentTraces[0] = currentTraces[1]
            currentTraces[1] = rg
            

            dc = sum(rg)/len(rg)
            print("DC: "+str(dc)+"V")
            dataArray.append(dict(Vg=Vg, Vd=Vd, Isd_mA=dc/Rfeedback))


    cleanupAnalogDiscovery(hdwf)



if __name__ == "__main__":
    font = "Helvetica 18"
    sg.set_options(font=font)
    text_size = (10, 1)
    big_button = (8,2)
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
    

    figWidth=870
    figHeight=350

    # First choose experiment:

    # Second, do everything else...



    transfer_curves = sg.Col([[sg.Text("Gate Voltage Vg", size=(3*(text_size[0]+param_size[0]),1), 
                    justification='left')],
                voltages("Vgate"),
                [sg.Text("Drain Voltage Vd", size=(3*(text_size[0]+param_size[0]),1), 
                    justification='left')],
                voltages("Vdrain", Npts=21),
                [Text("Gate delay (s):"), Input(default_text=f"2", key='-gate-delay-'),
                 Text("Drain delay (s):"), Input(default_text=f"0.2", key='-drain-delay-')]])
                 
    data_logger = sg.Col([[sg.Text("Gate Voltage Vg (V)", size=(2*(text_size[0]+param_size[0]),1), 
                    justification='left'), Input(default_text="0", key='-data_logger-Vg-')],
                [sg.Text("Drain Voltage Vd (V)", size=(2*(text_size[0]+param_size[0]),1), 
                    justification='left'), Input(default_text="0", key='-data_logger-Vd-')],
                    [sg.B("Update voltages", key='-data_logger-update-Voltages-', enable_events=True, size=text_size)]
                    ],
                    visible=False)


    options = [('transfer_curves', transfer_curves), ('data_logger', data_logger)]

    expt_keys = [x[0] for x in options]
    cols = [x[1] for x in options]

    # for key, c in options:
    #     c.update(visible=key==expt_keys[0])

    # Can this be modular?

    layout = [
            [sg.Text("Rfeedback (kOhm):", size=(15, 1)), Input(default_text='1.0', key='-Rfeedback-'),
             sg.Text("Frequency (kHz)", size=(10, 1)), Input(default_text="100", key='-frequency-kHz-'),
             sg.Text("Buffer", size=(8,1)), Input(default_text="1000", key='-buffer-')],
                [sg.Combo(expt_keys, default_value=expt_keys[0], key='-CHOOSE-EXPT-', enable_events=True)],
                cols,
                [sg.Text('AD connected: False', key="-devices-", size=(16,1))],
                [sg.Button("Run", size=big_button),
                sg.FileSaveAs("Save", target='-FILENAME-', default_extension="", size=big_button),
                sg.Input(visible=False, enable_events=True, key='-FILENAME-'),
                sg.Text("Data points: ", key='-npts-', size=(12,1)),
                sg.Button('Stop', size=big_button),
                sg.Button('Exit', size=big_button)
                ],
                [sg.Canvas(size=(figWidth, figHeight), background_color="white", key='-CANVAS-')]
                ]
    window = sg.Window("Analog Discovery IV curves", layout, finalize=True, size=(figWidth+40, 820))


    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas


    x1 = x2 = np.arange(bufferSize)/frequency*1000
    y1 = y2 = np.zeros(bufferSize)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(figWidth/72, figHeight/72))
    fig.subplots_adjust(left=0.15, wspace=0.35, right=1-0.02, top=1-0.03)
    ax1.grid(True)
    ax2.grid(True)

    expts = Experiments(fig, (ax1, ax2))

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    l1, = ax1.plot(x1, y1, '.')
    l2, = ax1.plot(x2, y2, '.')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (mA)")
    ax2.set_xlabel("Vdrain (V)")
    ax2.set_ylabel("$I_\\mathrm{sd}$ (mA)")
    # fig.tight_layout()
    # fig.subplots_adjust()
    
    fig_agg = draw_figure(canvas, fig)


    def update_plot(fig, ax1, ax2):
        ax1.relim()
        ax1.autoscale_view(tight=True)
        ax2.autoscale()
        ax2.autoscale_view(tight=True)
        fig.canvas.draw()
        fig.canvas.flush_events()


    class Thread:
        def is_alive(self):
            return False
    
    thread = Thread()
    was_running = False
    initialize = True
    done_initializing = False
    i = 0 #
    while True:
        i += 1
        event, values = window.read(timeout=100)
        running = thread.is_alive()
        if i % 100 == 0:
            print(f'{i=} {was_running=} {initialize=} {running=} {done_initializing=}')
        # Handle all of the possible events:
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            if thread.is_alive():
                exit_event.set()
                time.sleep(2) # Give the other thread time to complete?
            else:
                expts.tearDown()
            break
        
        if event == '-CHOOSE-EXPT-':
            for key, c in options:
                curr_expt = values['-CHOOSE-EXPT-']
                c.update(visible=key==curr_expt)

        try:
            Rfeedback = float(values['-Rfeedback-'])
        except:
            sg.Popup("Rfeedback must be a number.")
            window['-Rfeedback-'].update(str(Rfeedback))


        # Initialization routine -----
        if initialize:
            window.perform_long_operation(expts.setUp, '-AD-DEVICES-')
            initialize = False

        if event == '-AD-DEVICES-':
            window['-devices-'].update(f"AD connected: {values[event]}")
            done_initializing = True
        
        if not done_initializing:
            continue
        # End Initialization routine -----
        
        if event == 'Stop':
            print("Send stop signal")
            stop_event.set() # Tell thread to stop!

        if event == 'Run' and not running:
            data = []
            currentTraces = [None, None]
            prevPts = 0
            running = True
            Vgate = getVoltageInfo(values, "Vgate")
            Vdrain = getVoltageInfo(values, "Vdrain")
            Vg = float(values['-data_logger-Vg-'])
            Vd = float(values['-data_logger-Vd-'])
            buffer = int(values['-buffer-'])
            frequency = float(values['-frequency-kHz-']) * 1e3
            drain_delay = float(values['-drain-delay-'])
            gate_delay = float(values['-gate-delay-'])
            # Set up the thread to run the experiment!
            if values['-CHOOSE-EXPT-'] == 'transfer_curves':
                thread = threading.Thread(target=expts.transferCurves,
                args=(Vgate, Vdrain, Rfeedback, "f1", data, gate_delay,
                      drain_delay, currentTraces), daemon=False)
            else:
                thread = threading.Thread(target=expts.data_logger,
                args=(Vg, Vd, Rfeedback, frequency, buffer, data, currentTraces, window), daemon=False)
                ax2.set_xlabel("Time (s)")
            thread.start()
        
        if was_running:
            pts = len(data)
            window['-npts-'].update(f"Data points: {pts}")
            y1, y2 = currentTraces
            if y1 is not None:
                l1.set_ydata(y1)
            if y2 is not None:
                l2.set_ydata(y2)
    
            for point in data[prevPts:]:
                index = np.argmin(abs(point['Vg'] - Vgate))
                ax2.scatter(point[expts.x_col], point[expts.y_col], label=point['Vg'], c=cycle[index % 10])


            prevPts = pts

            # Do this every cycle?
            update_plot(fig, ax1, ax2)
            fig_agg.draw()

            
        
        if was_running and not running and len(data) > 0:
            df = pd.DataFrame(data)
            df['Vg_str'] = [f"{x:.3f}" for x in df['Vg']]
            df['Vd_str'] = [f"{x:.3f}" for x in df['Vd']]

            # if values['-CHOOSE-EXPT-'] == 'transfer_curves':
            #     fig_px = px.scatter(df, x='Vg', y='Isd_mA', color='Vd_str')
            #     fig_px.show()
            # else:
            #     fig_px = px.scatter(df, x='t_s', y='Isd_mA', color='Vd_str')
            #     fig_px.show()

            for point in data[prevPts:]:
                index = np.argmin(abs(point['Vg'] - Vgate))
                ax2.scatter(point[expts.x_col], point[expts.y_col], label=point['Vg'], c=cycle[index % 10])
            
            update_plot(fig, ax1, ax2)
            fig_agg.draw()

        if event == '-data_logger-update-Voltages-':
            change_voltage_event.set()
            print("Update voltages!")
        
        if event == '-FILENAME-':
            print("Handled filename event")
            if not values['-FILENAME-']:
                sg.popup("WARNING: File not saved.")
            else:
                df = pd.DataFrame(data)
                basename = make_filename(values, '-FILENAME-')
                output_filename = f'{basename}.xlsx'
                df.to_excel(output_filename)
                sg.popup(f"File saved to {output_filename}")
        
        was_running = running

        
    # main()