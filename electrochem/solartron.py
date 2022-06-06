import pyvisa
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import os
from munch import Munch
import binascii
import time

def random_key(size=8):
    return str(binascii.b2a_base64(os.urandom(6), newline=False))


def list_gpib_instruments():
    rm = pyvisa.ResourceManager()
    s = rm.list_resources()
    return rm, s


class AnalogDiscovery:
    def __init__(self):
        pass
    

    def layout(self):
        self.layout = [
            sg.Text('Analog Discovery')
        ]
        return self.layout
    
    def update(self, window, values):
        pass
    
    def handle_events(self, event, window, values):
        pass

class GPIBController:
    def __init__(self, title="Solartron Connection", gpib_address=4):
        self.title = title
        self.connected = False 
        self.gpib_address = gpib_address
        self.setUp()

    
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
        self.layout = [
            [sg.Text(self.title)],
            [sg.Text('Instruments:'), sg.Combo(self.resources, size=(14, 1), key='--GPIB-LIST--',  default_value=combo_default)],
            [sg.Text(self.status_text(), size=(20, 1), key='--GPIB-STATUS--')],
            [sg.Button('Connect', key='--GPIB-CONNECT--')]
        ]
        return self.layout
    
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
        self.layout = [
            [sg.Text(f'{self.title}')],
            [sg.Text('Time (s)', size=(10, 1)), sg.InputText("", size=(10, 1), key=self.time_key)],
            [sg.Button('Start', key=self.start_key), sg.Button('Stop', key=self.stop_key), sg.Button('Reset', key=self.reset_key)],
        ]
        return self.layout
    
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
        
        self.layout[1][1].update(f'{self.elapsed_time:.2f}')

    def __repr__(self):
        return f"Timer(title='{self.title}', id='{self.id}')"



def main(test=False):

    font = "Helvetica 18"
    sg.set_options(font=font, )

    text_size = (14, 1)
    param_size = (8, 1)

    # Components...
    
    c = Munch(ad=AnalogDiscovery(), gpib = GPIBController(), timer = Timer(title='Timer', kwargs={'size': text_size}))


    layout = [
    [c.ad.layout(), c.gpib.layout()],
    [c.timer.layout()],
    [sg.Button('Exit', size=(8,2))]
    ]

    window = sg.Window("Solartron Electrochemistry", layout, finalize=True, size=(900, 700))


    while True:
        event, values = window.read(timeout=100)

                # Handle all of the possible events:
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        
        for val in c.values():
            val.handle_events(event, window, values)
            val.update(window, values)


if __name__ == '__main__':
    main()