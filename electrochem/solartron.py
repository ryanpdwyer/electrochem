import pyvisa
import os
import datetime
import numpy
np = numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import os
import signal
from munch import Munch
import binascii
from ctypes import *
import time
import threading
from cvnew import CV, RVMock

from dwfconstants import *
import sys

exit_event = threading.Event()
stop_event = threading.Event()
change_voltage_event = threading.Event()


if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

def random_key(size=8):
    return str(binascii.b2a_base64(os.urandom(6), newline=False))


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

def list_gpib_instruments():
    rm = pyvisa.ResourceManager()
    s = rm.list_resources()
    return rm, s


class CVGUI:
    def __init__(self, rv=None):
        self.rv = rv
        self.connected = False
        self.cv = None
        self.setUp()
        self.modes = ['OCP', 'Setup EIS', 'EIS', 'CV', 'CV+EIS']
    
    def setUp(self):
        if self.rv is not None:
            self.cv = CV(self.rv)
            self.connected = True
            print("Connected CV to Solartron")
        else:
            print("No GPIB connection - not connected")
    
    def tearDown(self):
        pass
    
    def log(self):
        if self.cv is not None:
            return "\n".join([repr(x) for x in self.cv.log])
        else:
            return ""

    def layout(self):
        self.output_options = {'x1': 1, 'x10': 10}
        self.input_options = {'x1': 1, 'x0.01': 0.01}
        self.control_bandwidths = {'600 kHz': 0, '360 kHz': 1, '> 1 MHz': 2, 
        '> 600 kHz': 3, '24 kHz': 4, '8 kHz': 5, '2.4 kHz': 6, '800 Hz': 7,
        '80 Hz': 8, '8 Hz': 9}
        self.output_options_keys = list(self.output_options.keys())
        self.input_options_keys = list(self.input_options.keys())
        self.control_bandwidths_keys = list(self.control_bandwidths.keys())
        current_ranges = list(self.cv.current_ranges_dict.keys())
        current_lims = list(self.cv.current_limits_dict.keys())

        self.expt_params = {'OCP': sg.Col([[sg.Text('Points'), sg.Input(10, key='--CV-OCP-points--')]], visible=False),
                        'Setup EIS': sg.Col([[sg.Text('Amp. (V)'), sg.Input(1, key='--CV-Setup-EIS-Amp--'),
                                            sg.Text('Freq. (Hz)'), sg.Input(100, key='--CV-Setup-EIS-Freq--')
                                            ],
                                            [sg.Text('Ch. 1 Range'), sg.Combo(['200 mV', '2.5 V'], default_value = '2.5 V', key='--CV-EIS-Ch1--'),
                                            sg.Text('Ch. 2 Range'), sg.Combo(['200 mV', '2.5 V'], default_value = '2.5 V', key='--CV-EIS-Ch2--')]], visible=True)}

        self._layout = [
            [sg.T("Current Range"), sg.Combo(current_ranges, default_value='2 mA', key='--CV-IRANGE--'),
            sg.T('Current Lim'), sg.Combo(current_lims, default_value='2 mA', key='--CV-ILIM--'),
            sg.T('Potential (V)'), sg.Input(default_text='0', key='--CV-Potential--')
            ],
            [sg.T("Volt. Out Gain"),
            sg.Combo(self.output_options_keys, default_value=self.output_options_keys[0], key='--CV-VoltageOutGain--'),
            sg.T("Curr. Out Gain"),
            sg.Combo(self.output_options_keys, default_value=self.output_options_keys[0], key='--CV-CurrentOutGain--'),
            sg.T("Volt. Input Gain"),
            sg.Combo(self.input_options_keys, default_value=self.input_options_keys[0],
                key='--CV-VoltageInGain--')],
            [sg.T("Control Loop Bandwidth"),
            sg.Combo(self.control_bandwidths_keys, default_value=self.control_bandwidths_keys[2])],
            [sg.Button("Zero Offset"), sg.Button("Update Settings"), sg.T("Update needed")],
            [sg.T('Choose Mode:'), sg.Combo(self.modes, self.modes[0], key='--CV-Mode--', enable_events=True)],
            self.expt_params.values(),
            [sg.Multiline(self.log(), key='--CV-GUI-LOG--')]]
        return self._layout

    def handle_events(self, event, window, values):
        self.handlers = {
        }
        if event in self.handlers:
            self.handlers[event](window, values)
    
    def update(self, window, values):
        self._layout[-1][0].update(self.log())




class AnalogDiscovery:
    def __init__(self):
        self.hdwf = None
        self.info = []
    

    def layout(self):
        self._layout = [
        [sg.Text(self.connection_status(), key='--AnalogDiscovery-Status--')],
        [sg.Button('Connect', key='--AnalogDiscovery-Connect--')]
        ]
        return self._layout
    
    def update(self, window, values):
        pass
    
    def connect(self, window, values):
        hdwf, info = setupAnalogDiscovery()
        self.hdwf = hdwf
        self.info.append(info)
        print(info)
        window['--AnalogDiscovery-Status--'].update(self.connection_status())
    
    def connection_status(self):
        if self.hdwf is None:
            conn_str = 'Not connected'
        else:
            conn_str = 'Connected'
        return f"Status: {conn_str}"
    
    def handle_events(self, event, window, values):
        self.handlers = {
            '--AnalogDiscovery-Connect--': self.connect
        }
        if event in self.handlers:
            self.handlers[event](window, values)

class GPIBController:
    def __init__(self, gpib_address=4, test=False):
        self.connected = False 
        self.gpib_address = gpib_address
        self.rv = None
        self.setUp()
        if test:
            self.rv = RVMock()
            self.connected=True

    
    def setUp(self):
        self.rm = pyvisa.ResourceManager()
        self.resources = self.rm.list_resources()
        # self.default_index = 0
        expected_resource = "GPIB0::" + str(self.gpib_address) + "::INSTR"
        self.resource_index = None
        self.connect_to_resource(expected_resource)


    def connect_to_resource(self, resource_name):
        if len(self.resources) > 0:
            if resource_name in self.resources:
                self.resource_index = list(self.resources).index(resource_name)
            else:
                self.resource_index = 0

        if self.resource_index is not None:
            self.rv = self.rm.open_resource(self.resources[self.resource_index])
            self.connected = True
            # Need another parameter here - what am I connected to...

    def status_text(self):
        connected_text = 'Connected' if self.connected else "Disconnected"
        status = f'Status: {connected_text}'
        return status

    def layout(self):
        combo_default = None
        if self.resource_index is not None:
            combo_default = self.resources[self.resource_index]
        self._layout = [
            [sg.Text('Instruments:'), sg.Combo(self.resources, size=(14, 1), key='--GPIB-LIST--',  default_value=combo_default)],
            [sg.Text(self.status_text(), size=(20, 1), key='--GPIB-STATUS--')],
            [sg.Button('Connect', key='--GPIB-CONNECT--')]
        ]
        return self._layout
    
    def connect(self, window, values):
        selected_instrument = values['--GPIB-LIST--']
        self.connect_to_resource(selected_instrument)
        window['--GPIB-STATUS--'].update(self.status_text())


    def handle_events(self, event, window, values):
        self.handlers = {
            '--GPIB-CONNECT--':  self.connect
        }
        if event in self.handlers:
            self.handlers[event](window, values)

    def tearDown(self):
        pass
    
    def update(self, window, values):
        "No code that needs to run on every window update?"
        pass
    




class Timer:
    def __init__(self, title='Timer', **kwargs):
        self.title = 'Timer'
        self.kwargs = kwargs
        self.id=random_key()
        self.running = False
        self.carryover_time = 0
    

    def layout(self):

        self.time_key = f'--Timer-time-'+self.id
        self.start_key = f'--Timer-start-'+self.id
        self.stop_key = f'--Timer-stop-'+self.id
        self.reset_key = f'--Timer-reset-'+self.id
        self._layout = [
            [sg.Text(f'{self.title}')],
            [sg.Text('Time (s)', size=(10, 1)), sg.InputText("", size=(10, 1), key=self.time_key)],
            [sg.Button('Start', key=self.start_key), sg.Button('Stop', key=self.stop_key), sg.Button('Reset', key=self.reset_key)],
        ]
        return self._layout
    
    # Event handlers...
    def start(self, window, values):
        self.start_time = time.time()
        self.running = True
    
    def stop(self, window, values):
        if self.running:
            self.carryover_time += time.time() - self.start_time
        self.running = False

    def reset(self, window, values):
        self.running = False
        self.carryover_time = 0

    def handle_events(self, event, window, values):
        self.handlers = {
            self.start_key: self.start,
            self.stop_key: self.stop,
            self.reset_key: self.reset,
        }
        if event in self.handlers:
            self.handlers[event](window, values)

    def update(self, window, values):
        if self.running:
            self.elapsed_time = time.time() - self.start_time + self.carryover_time
        else:
            self.elapsed_time = self.carryover_time
        
        self._layout[1][1].update(f'{self.elapsed_time:.2f}')

    def __repr__(self):
        return f"Timer(title='{self.title}', id='{self.id}')"



def main(test=False):

    font = "Helvetica 18"
    sg.set_options(font=font, )

    text_size = (14, 1)
    param_size = (8, 1)

    # Components...
    
    c = Munch(ad=AnalogDiscovery(), gpib = GPIBController(test=test))

    print(c.gpib.rv)
    c.cvGUI = CVGUI(rv=c.gpib.rv)
    print(c.cvGUI.cv.query("?VN"))




    layout = [
    [sg.TabGroup([[
    sg.Tab('GPIB', c.gpib.layout()),
    sg.Tab('AD', c.ad.layout()),
    sg.Tab("Solartron", c.cvGUI.layout()),
    sg.Tab('Output', [[sg.Output((80, 3))]])]])],
    [sg.Button('Exit', size=(8,2))]
    ]

    window = sg.Window("Solartron Electrochemistry", layout, finalize=True, size=(1100, 700))


    while True:
        event, values = window.read(timeout=100)

                # Handle all of the possible events:
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        
        for val in c.values():
            val.handle_events(event, window, values)
            val.update(window, values)
        
        if test:
            if event != '__TIMEOUT__':
                print(event)


if __name__ == '__main__':
    import sys
    args = sys.argv
    if args[-1] == 'test':
        test = True
    else:
        test = False
    main(test=test)