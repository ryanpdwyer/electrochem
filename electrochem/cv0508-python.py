import pyvisa
import os
from json_tricks import dump, dumps
import numpy as np
from datetime import datetime
from numpy.random import rand
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def float_or_int(x):
    try:
        return int(x)
    except ValueError:
        return float(x)

class RVMock:
    def __init__(self):
        self.params = dict()
        self.rate = 13.0
        self.t0 = None

    def write(self, message):
        setting, value = message[:2], float_or_int(message[2:])
        self.params[setting] = value
        if message == "SW1":
            self.t0 = time.time() # Start sweeping!
            self.t = 0

    @property
    def times(self):
        return np.array([self.params[k] for k in ['TA', "TB", "TC", "TD"]])


    @property
    def segments(self):
        return self.params['SM']
    
    @property
    def ttotal(self):
        return sum(self.times[i % 4] for i in range(self.segments))

    def V(self, t):
        if t > self.ttotal:
            return self.params['VD']
        else:
            t_used = t % sum(self.times) # Position in the cycle, after this length we repeat...
            if t_used < self.params['TA']:
                return self.params['VA'] + (self.params['VB'] - self.params['VA']) * t_used / self.params['TA']
            elif t_used < (self.params['TA'] + self.params['TB']):
                return self.params['VB'] + (self.params['VC'] - self.params['VB']) * (t_used - self.params['TA']) / self.params['TB']
            elif t_used < (self.params['TA'] + self.params['TB'] + self.params['TC']):
                return self.params['VC'] + (self.params['VD'] - self.params['VC']) * (t_used - self.params['TA'] - self.params['TB']) / self.params['TC']
            else:
                return self.params['VD'] + (self.params['VA'] - self.params['VD']) * (t_used - self.params['TA'] - self.params['TB'] - self.params['TC']) / self.params['TD']


    def I(self, t):
        return self.V(t) / 12.0e6 # Fixed resistance

    def read_ascii_values(self, timeout=7000, container=list):
        dt = 1.0/self.rate
        self.tprev = self.t
        while True:
            self.t = time.time() - self.t0 # Current time...
            if self.t > self.ttotal:
                time.sleep(timeout/1000)
                raise ValueError
            elif self.t < (self.tprev + dt):
                # print(f"{self.t} Sleeping")
                # print(f"{self.tprev}")
                time.sleep(0.03)
            else:
                # print("Escaped")
                return container([self.V(self.t), np.random.randn()*1e-9+self.I(self.t),
                                0, 0, 1, np.floor(self.t / 60), np.floor(self.t % 60), int((self.t % 1)*100)])
        



class CV:
    def __init__(self, rv):
        self.rv = rv
        self.rv.timeout = 7000
        self.rv.write_termination = "\n"
        self.log = []
        self.ocp = [] # Record of all OCP measurements performed...
        self.expt = 0
        self.data = {}
        # The current dataset goes here...
        self.current_data = []
        self.sweep_mode = "SW0"
        self.ix = 0 
        self.iy = 1

        # CV knows the sweep mode - maybe it should know the default x and y
        # variables to plot as well


    def write(self, message):
        self.log.append((message, self.rv.write(message)))
    

    
    def measure_ocp(self, Npts=10):
        
        self.ix = -1
        self.iy = 0
        self.xlabel = "Time (s)"
        self.ylabel = "OCP vs. Ref (V)" # Set up everything needed to plot?

        self.write("PO0") # Set mode to potentiostat
        self.write("PW0") # Turn polarization OFF
        self.write("ON1") # Stay in rest mode - do not actually apply a voltage!
        self.write("SW0") # Sweep off...



        self.setup_voltammetry_outputs() # Default X = ΔRV, Y = I.

        # DVM setup
        self.write("DG2") # Number of digits: 0 = 5, 2 = 4x9, 60 Hz
        self.write("RG0") # Autorange voltage input range = 0, or use RG2 for the 2V mode...
        
        

        self.write("TR1") # Continuous measurement mode
        self.write("DC1") # Drift correction OFF
        self.write("AV0") # Averaging OFF
        self.write("NU0") # Null OFF

        # Now we are set up, we just need to run the measurement...
        # time.sleep(1) # Wait a bit, then run the DVM

        self.current_data = []

        self.write("RU1") # Turn the DVM ON
        time.sleep(0.05)
        for i in range(Npts):
            try:
                data_pt = self.get_data()
                print(data_pt)
                self.ocp.append(data_pt)
                self.current_data.append(data_pt)
                # Could need a little delay cushion in here?
            except:
                print("Expt done?")
                pass
                break
    
        self.write("RU0")
        self.write("PW0") # Turn polarization off
        self.write("BK0") # Do a reset, which seems like a generally good idea
    
        return np.array(self.current_data)



        
    
    def setup_voltammetry_outputs(self):
        self.write("PX3") 
        self.write("PY5") 
        self.write("UL3") # Display Delta RE on left
        self.write("UR5") # Display current on right

    def setup_analog_sq_wave_voltammetry(self, V1, V2, T1, T2, segments=4, Irange=0, Ilim=3):
        voltages = np.array([V1, V1, V2, V2, V1])
        dV = abs(V2-V1)
        rate = 100 # V/s...
        fast_step_time = max(dV/rate, 0.01) # Minimum 10 ms time step...
        times = np.array([T1, fast_step_time, T2, fast_step_time])
        
        #######################################################
        self.write("PW0") # Turn polarization OFF
        self.write("RU0") # Turn the DVM OFF
        self.write("SW0")
        
        # This is awkward - this should probably be done in run...
        # Each experiment should be autolabeled
        # Setup should just do setup, and save this information to a
        # self.current_setup dictionary?
        # Should I have one giant dataclass for the Solartron State? Possibly...
        self.expt += 1
        self.data[self.expt] = dict()
        self.data[self.expt]['params'] = dict(experiment='Sq Wave Voltammetry - Analog Sweep', V1=V1, V2=V2, T1=T1, T2=T2, segments=segments, Irange=Irange, Ilim=Ilim)
        
        
        # Polarization / Mode
        self.write("PO0") # Set mode to potentiostat
        self.write(f"PV{V1}") # Set polarization to initial value
        self.write("ON0") # Set to Pol I/V mode, will go to initial polarization ON
        
        
        # Current sense setup
        self.write(f"RR{Irange}") # Auto-range is 0
        if Irange == 0: # If auto-ranging, then we set the current limit here...
            self.write(f"IL{Ilim}") # 2 \times 10^{Ilim} uA (0 = 2 uA, 3 = 2 mA, 6 = 2 A)
        self.write("OL1") # Limit the current (0 = cutout, 2 = No Limit = BAD)
        
        
        # Compensation
        self.write("CC0") # IR Compensation OFF
        
        
        # DVM setup
        self.write("DG2") # Number of digits: 0 = 5, 2 = 4x9, 60 Hz
        self.write("RG0") # Autorange voltage input range = 0, or use RG2 for the 2V mode...
        
        self.write("TR3") # Sweep synchronized measurements
        self.write("DC1") # Drift correction OFF
        self.write("AV0") # Averaging OFF
        self.write("NU0") # Null OFF
        
        
        # Conditioning
        self.write("BR0") # Bias reject OFF
        self.write("FI0") # 10 Hz filter OFF
        
        # Output to screen and GPIB setup
        self.setup_voltammetry_outputs()
        self.write("GP1") # Long output to GPIB ON
        
        #######################3
        #### Setup Sweep
        ########################
        self.write(f"SM{segments}") # Segments = 4
        self.write("DL1") # Delay 1 second before starting

        self.sweep_mode = "SW1" # Use the analog sweep when we actually run  
        
        # Set times
        for t, v, name in zip(times, voltages, ["A", "B", "C", "D"]):
            self.write(f"V{name}{v}") 
            self.write(f"T{name}{t}")

    
    
    def setup_cv(self, V1, V2, V3, V4, rate, segments=4, Irange=0, Ilim=3):
        voltages = np.array([V1, V2, V3, V4, V1])
        dVoltage = np.abs(np.diff(voltages))
        times = dVoltage / rate
        
        #######################################################
        self.write("PW0") # Turn polarization OFF
        self.write("RU0") # Turn the DVM OFF
        self.write("SW0")
        
        self.expt += 1
        self.data[self.expt] = dict()
        self.data[self.expt]['params'] = dict(experiment='Cyclic Voltammetry', V1=V1, V2=V2, V3=V3, V4=V4, rate=rate, segments=segments, Irange=Irange, Ilim=Ilim)
        
        
        # Polarization / Mode
        self.write("PO0") # Set mode to potentiostat
        self.write(f"PV{V1}") # Set polarization to initial value
        self.write("ON0") # Set to Pol I/V mode, will go to initial polarization ON
        
        
        # Current sense setup
        self.write(f"RR{Irange}") # Auto-range is 0
        if Irange == 0:
            self.write(f"IL{Ilim}") # 2 \times 10^{Ilim} uA (0 = 2 uA, 3 = 2 mA, 6 = 2 A)
        self.write("OL1") # Limit the current (0 = cutout, 2 = No Limit = BAD)
        
        
        # Compensation
        self.write("CC0") # IR Compensation OFF
        
        
        # DVM setup
        self.write("DG2") # Number of digits: 0 = 5, 2 = 4x9, 60 Hz
        self.write("RG0") # Autorange voltage input range = 0, or use RG2 for the 2V mode...
        
        self.write("TR3") # Sweep synchronized measurements
        self.write("DC1") # Drift correction OFF
        self.write("AV0") # Averaging OFF
        self.write("NU0") # Null OFF
        
        
        # Conditioning
        self.write("BR0") # Bias reject OFF
        self.write("FI0") # 10 Hz filter OFF
        
        # Output to screen and GPIB setup
        self.write("PX3") 
        self.write("PY5") 
        self.write("UL3") # Display Delta RE on left
        self.write("UR5") # Display current on right
        self.write("GP1") # Long output to GPIB ON
        
        #######################
        #### Setup Sweep
        ########################
        self.write(f"SM{segments}") # Segments = 4
        self.write("DL1") # Delay 1 second before starting

        self.sweep_mode = "SW1" # Use the analog sweep when we actually run the CV 
        
        # Set voltages
        self.write(f"VA{V1}")
        self.write(f"VB{V2}")
        self.write(f"VC{V3}")
        self.write(f"VD{V4}")
        # Set times
        for t, name in zip(times, ["TA", "TB", "TC", "TD"]):
            self.write(f"{name}{t}")
        

    def get_data(self):
        """Return the data from the potentiostat in the format

            xdata, ydata, xerr, yerr, time(s)

        """

        data = self.rv.read_ascii_values(container=np.array)
        return np.r_[data[:4], np.dot(data[4:], np.array([3600, 60, 1, 0.01]))]
        

    def run(self):
        # What else do we need to know?
        # Which two columns are the x and y data to plot?
        # 
        self.write("PW1") # Turn polarization ON
        self.write("RU1") # Turn the DVM ON
        self.write(self.sweep_mode) # Start the (analog) sweep!
        
        self.current_data = []
        
        try:
            while True:
                self.current_data.append(self.get_data())
        except:
            print("Expt done?")
            pass
        
        
        self.data[self.expt]['data'] = np.array(self.current_data)
        self.write("PW0")
        self.write("RU0")
        self.write("SW0")
        self.write("BK0")
        return np.array(self.current_data)

    def stop(self):
        self.current_data = []
        self.write("PW0")
        self.write("RU0")
        self.write("SW0")
        self.write("BK0") # Break, resetting everything!
        print("Stopped!")    


current_ranges_dict = {"Auto": 0, "2 uA": 1, "20 uA": 2, "200 uA": 3, '2 mA': 4, '20 mA': 5, '200 mA': 6, '2 A': 7}
current_limits_dict = {"2 uA": 0, "20 uA": 1, "200 uA": 2, '2 mA': 3, '20 mA': 4, '200 mA': 5, '2 A': 6}

def startCV(cv, values, window):
    cv.setup_cv(float(values['-CV-V1']),
            float(values['-CV-V2']),
            float(values['-CV-V3']),
            float(values['-CV-V4']),
                float(values['-CV-scan_rate'])/1000, # mV/s to V/s
                segments=int(float(values['-CV-cycles'])*4),
                Irange=current_ranges_dict[values['current_range']],
                Ilim=current_limits_dict[values['current_limit']]
                )
    time.sleep(0.05)
    cv.run()
    window.write_event_value('-THREAD-', 'donnnnnne')  # put a message into queue for GUI

def startSqWaveV(cv, values, window):
    cv.setup_analog_sq_wave_voltammetry(float(values['-SQ-V1']),
            float(values['-SQ-V2']),
            float(values['-SQ-T1']),
            float(values['-SQ-T2']),
                segments=int(float(values['-SQ-cycles'])*4),
                # These current settings are in theory independent of some of
                # the others - could reuse these...
                Irange=current_ranges_dict[values['current_range']],
                Ilim=current_limits_dict[values['current_limit']]
                )
    time.sleep(0.05)
    cv.run()
    window.write_event_value('-THREAD-', 'donnnnnne')  # put a message into queue for GUI

def startOCP(cv, values, window):
    cv.measure_ocp(int(values['-OCP-Npts']))
    time.sleep(0.05)
    window.write_event_value('-THREAD-', 'donnnnnne')  # put a message into queue for GUI


start_expt_dict = {'Cyclic Voltammetry': startCV, 'Sq Wave Voltammetry': startSqWaveV, 'OCP': startOCP}
expt_plots_dict = {'Cyclic Voltammetry': dict(ix=0, iy=1, yscale=1e3, xlabel="Potential vs. Ref (V)", ylabel='Current (mA)'),
'Sq Wave Voltammetry': dict(ix=-1, iy=1, xlabel="Time (s)", yscale=1e3, ylabel='Current (mA)'),
                    'OCP': dict(ix=-1, iy=0, ylabel="OCP vs. Ref (V)", xlabel="Time (s)", yscale=1)}

def list_keys(d):
    return list(d.keys())


def make_filename(cv, values, key):     
    today = datetime.today()
    datestring = today.strftime("%Y-%m-%d %H-%M")
    directory, basefilename = os.path.split(values[key])
    return os.path.join(directory, f"{datestring} {basefilename}")

def main(test=False):

    font = "Helvetica 18"
    sg.set_options(font=font)

    if test:
        rv = RVMock()
    else:
        rm = pyvisa.ResourceManager()
        s = rm.list_resources()

        rv = rm.open_resource(s[0])
    cv = CV(rv)

    text_size = (14, 1)
    param_size = (8, 1)
    # define the form layout

    current_ranges = list_keys(current_ranges_dict)
    current_limits = list_keys(current_limits_dict)

    col_CV = [[sg.Text('V1 (Start)', size=text_size), sg.Input(key="-CV-V1", size=param_size),
               sg.Text('V2', size=text_size), sg.Input(key="-CV-V2", size=param_size)],
                [sg.Text('V3', size=text_size), sg.Input(key="-CV-V3", size=param_size),
                sg.Text('V4', size=text_size), sg.Input(key="-CV-V4", size=param_size)],
                [sg.Text('Scan rate (mV/s)', size=text_size), sg.Input(key="-CV-scan_rate", size=param_size),
                sg.Text('Cycles', size=text_size), sg.Input(key="-CV-cycles", size=param_size)]
                ]

    col_Sq = [[sg.Text('V1 (Start)', size=text_size), sg.Input(key="-SQ-V1", size=param_size),
               sg.Text('V2', size=text_size), sg.Input(key="-SQ-V2", size=param_size)],
                [sg.Text('T1', size=text_size), sg.Input(key="-SQ-T1", size=param_size),
                sg.Text('T2', size=text_size), sg.Input(key="-SQ-T2", size=param_size)],
                [sg.Text('Cycles', size=text_size), sg.Input(key="-SQ-cycles", size=param_size)]
                ]

    col_OCP = [[sg.Text('NPts', size=text_size), sg.Input(key="-OCP-Npts", default_text="25", size=param_size),]]
    
    experiment_dict = {'Cyclic Voltammetry': col_CV, 'Sq Wave Voltammetry': col_Sq, 'OCP': col_OCP}
    columns = [sg.Column(val, k=key, visible=key=='Cyclic Voltammetry') for key, val in experiment_dict.items()]
    layout = [[sg.Combo(list_keys(experiment_dict),
                default_value='Cyclic Voltammetry', k='-SELECT_EXPERIMENT-',
                size=(25, 1), enable_events=True)],
                columns, # This is our key spot...
                [sg.Text("Current Range", size=text_size),
                sg.Combo(current_ranges, default_value=current_ranges[0],
                          key='current_range', size=param_size),
                sg.Text("Auto Current Limit", size=text_size),
                sg.Combo(current_limits, default_value=current_limits[3],
                          key='current_limit', size=param_size)],
                [sg.B("Start"),
                sg.B("Stop"),
                sg.FileSaveAs("Save", target='-FILENAME-', default_extension=""),
                sg.Input(visible=False, enable_events=True, key='-FILENAME-'), 
                sg.FileSaveAs("Save Image", target='-FILENAME-PNG-', default_extension=""),
                sg.Input(visible=False, enable_events=True, key='-FILENAME-PNG-'), 
                sg.Text("Status: ", key='status', size=(15, 1)),
                sg.Text("Expt: ", key='Expt', size=(9, 1))],
                [sg.Canvas(size=(640, 480), key='-CANVAS-')],
                [sg.Button('Exit', size=(10, 2), pad=((280, 0), 3)), sg.Text("Pts: ", key='npts', size=(10,2))]]

    window = sg.Window("Window Title", layout, finalize=True)

    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    # draw the intitial scatter plot
    fig, ax = plt.subplots()
    ax.grid(True)
    xdata = []
    ydata = []

    fig_agg = draw_figure(canvas, fig)
    lines, = ax.plot(xdata, ydata, '.')
    ax.set_xlabel("Potential vs. Ref (V)")
    ax.set_ylabel("Current (mA)")



    while True:
        event, values = window.read(timeout=100)

        # Handle all of the possible events:
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

        # When an experiment is selected, we need to update the plots and sliders:
        if event == '-SELECT_EXPERIMENT-':
            for key, col in zip(experiment_dict.keys(), columns):
                curr_expt  = values['-SELECT_EXPERIMENT-']
                col.update(visible=key==curr_expt)
                ax.set_xlabel(expt_plots_dict[curr_expt]['xlabel'])
                ax.set_ylabel(expt_plots_dict[curr_expt]['ylabel'])

        # 
        if event == 'Start' and 'Running' not in window['status'].get():
            # Reset the data on the screen...
            print("Starting!")
            xdata = []
            ydata = []
            tdata = []
            cv.current_data = []
            # This needs to be aware of what the value of [-SELECT_EXPERIMENT-] is
            # Launch the correct function  
            threading.Thread(target=start_expt_dict[values['-SELECT_EXPERIMENT-']],
                             args=(cv, values, window), daemon=True).start()

            window['status'].update('Status: Running')
            time.sleep(1) # Wait until the CV is actually running...
            window['Expt'].update(f"Expt: {cv.expt}")
        
        if event == 'Stop':
            cv.stop() # Probably a good idea?

        if event == '-FILENAME-':
            if not values['-FILENAME-']:
                sg.popup("File not saved.")
            else:
                basename = make_filename(cv, values, '-FILENAME-')
                output_filename = f'{basename}.json'
                with open(output_filename, 'w') as fh:
                    dump(cv.data, fh)
                sg.popup(f"File saved to {output_filename}.")
        
        if event == '-FILENAME-PNG-':
            if not values['-FILENAME-PNG-']:
                sg.popup("File not saved.")
            else:
                basename = make_filename(cv, values, '-FILENAME-PNG-')
                fig.savefig(f'{basename} Expt {cv.expt}.png', bbox_inches='tight', dpi=200)

            
        if event == '-THREAD-':
            window['status'].update('Status: Done')


        if 'Running' in window['status'].get():
            # TODO: Need a way to set which will be the x data and y data
            # This is WAY too much of a kludge
            pts = len(xdata)
            print(f"{pts} {len(cv.current_data)}")
            window['npts'].update(f'Pts: {pts}')
            # xdata.extend([x[expt_plots_dict[values['-SELECT_EXPERIMENT-']]['i']] for x in cv.current_data[pts:]])
            ## Assuming ydata is current in A - converting to mA
            # ydata.extend([x[1]*1e3 for x in cv.current_data[pts:]])

            e_plot_dict = expt_plots_dict[values['-SELECT_EXPERIMENT-']]
            current_data = np.array(cv.current_data)
            # print(current_data)
            if len(current_data) > 0:
                current_data[:, -1] -= current_data[0, -1]
                lines.set_xdata(current_data[:, e_plot_dict['ix']])
                lines.set_ydata(current_data[:, e_plot_dict['iy']]*e_plot_dict['yscale']) # To milliAmps

            ax.relim()
            ax.autoscale_view(tight=True)
            fig.canvas.draw()
            fig.canvas.flush_events()

            fig_agg.draw()


    window.close()

if __name__ == "__main__":
    import sys
    args = sys.argv
    if args[-1] == 'test':
        test = True
    else:
        test = False
    main(test=test)

