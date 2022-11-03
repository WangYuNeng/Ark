from math import sin
import matplotlib.pyplot as plt
import PySpice.Logging.Logging as Logging
import numpy as np
logger = Logging.setup_logging()


from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *

def render_transient(analysis, file_name, data_points):
    time_data = analysis.time.as_ndarray()
    for i, (name,data) in enumerate(analysis.nodes.items()):
        if name not in data_points:
            continue
        series_name = name
        series_data = data.as_ndarray()
        plt.plot(time_data,series_data, label=series_name)
        plt.legend()
    plt.title('theta={}'.format(file_name[4:-4]))
    plt.savefig(file_name)
    plt.clf()

def sim(circuit):
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=1e-11, end_time=75e-9)
                                
    return analysis

def addGmCSubCkt(circuit):
    circuit.raw_spice += '''
.subckt gmc vi1 vi2 vi3 vo Cint=1e-12 gm1=1e-3 gm2=-1e-3 gm3=1e-3 Rloss=1e12
Ccvo vo 0 c=Cint
Gvccs1 0 vo vi1 0 gm1
Gvccs2 0 vo vi2 0 gm2
Gvccs3 0 vo vi3 0 gm3
RR0 vo 0 r=Rloss
.ends gmc

'''

def addGmRSubCkt(circuit):
    circuit.raw_spice += '''
.subckt gmr vi1 vi2 vi3 vo gm1=1e-3 gm2=1e-3 gm3=1e-3 R=1e3
Gvccs1 0 vo vi1 0 gm1
Gvccs2 0 vo vi2 0 gm2
Gvccs3 0 vo vi3 0 gm3
RR0 vo 0 r=R
.ends gmr

'''

def wire_tline(circuit, in_name, out_name, delay):

    n_ladder = int(delay / 1e-9)
    def named_l(i):
        return 'l_{}_{}'.format(in_name, i)
    def named_c(i):
        return 'c_{}_{}'.format(in_name, i)

    circuit.X(named_c(0), 'gmc', in_name, named_l(0), 0, named_c(0))
    for i in range(n_ladder-1):
        circuit.X(named_l(i), 'gmc', named_c(i), named_c(i+1), 0, named_l(i))
        circuit.X(named_c(i+1), 'gmc', named_l(i), named_l(i+1), 0, named_c(i+1))
    i = n_ladder - 1
    circuit.X(named_l(i), 'gmc', named_c(i), out_name, 0, named_l(i))
    circuit.X(named_c(i+1), 'gmc', named_l(i), out_name, 0, out_name)


def wire_ckt(td, theta, d, c):
    circuit = Circuit('Beamformer')
    addGmCSubCkt(circuit=circuit)
    addGmRSubCkt(circuit=circuit)

    w = 10

    circuit.PulseVoltageSource('pulse1', 'in1', circuit.gnd, initial_value=0@u_V, pulsed_value=1@u_V, 
        pulse_width=w@u_ns, period=1000@u_us, delay_time=0, rise_time=w/2@u_ns, fall_time=w/2@u_ns)
    circuit.PulseVoltageSource('pulse2', 'in2', circuit.gnd, initial_value=0@u_V, pulsed_value=1@u_V, 
        pulse_width=w@u_ns, period=1000@u_us, delay_time=d*sin(theta)/c, rise_time=w/2@u_ns, fall_time=w/2@u_ns)
    circuit.PulseVoltageSource('pulse3', 'in3', circuit.gnd, initial_value=0@u_V, pulsed_value=1@u_V, 
        pulse_width=w@u_ns, period=1000@u_us, delay_time=2*d*sin(theta)/c, rise_time=w/2@u_ns, fall_time=w/2@u_ns)


    in1, in2, in3, out1, out2 = 'in1', 'in2', 'in3', 'out1', 'out2'
    wire_tline(circuit, in1, out1, delay=2*td)
    wire_tline(circuit, in2, out2, delay=td)
    # circuit.X('sum', 'gmr', out1, out2, in3, 'sum')
    circuit.X('sum', 'gmc', out1, out2, in3, 'sum', gm1=1e-3, gm2=1e-3, gm3=1e-3, Rloss=1e3, Cint=1e-15)

    print(circuit)
    return circuit
c, td, d = 3e8, 5e-9, 50e-1
print(c*td/d , np.arcsin(c*td/d))
for theta in np.arange(0, np.pi/2, 0.1):
    circuit = wire_ckt(td, theta, d, c)
    analysis = sim(circuit)
    render_transient(analysis, 'test{}.png'.format(theta), {'in1', 'in2', 'in3', 'out1', 'out2', 'sum'})