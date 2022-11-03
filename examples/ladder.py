from math import sin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import PySpice.Logging.Logging as Logging
import numpy as np
logger = Logging.setup_logging()


from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from saa.ladder_spec import *
# np.random.seed(0)
def render_transient(analysis, file_name, data_points):
    time_data = analysis.time.as_ndarray()
    for i, (name,data) in enumerate(analysis.nodes.items()):
        if name not in data_points:
            continue
        series_name = name
        series_data = data.as_ndarray()
        plt.plot(time_data,series_data, label=series_name)
        plt.legend()
    plt.savefig(file_name)
    plt.clf()

def sim(circuit):
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=1e-11, end_time=75e-9)
                                
    return analysis

def pulse(t, amplitude, delay, rise_time, fall_time, pulse_width, period):
    t = t % period - delay
    if rise_time <= t and pulse_width + rise_time >= t:
        return amplitude
    elif t < rise_time:
        return amplitude * t / rise_time
    elif pulse_width + rise_time < t and pulse_width + rise_time + fall_time >= t:
        return amplitude * (1 - (t - pulse_width - rise_time) / fall_time)
    return 0

def create_graph():
    lg = LadderGraph()
    params = {
        'amplitude': 1,
        'delay': 0,
        'rise_time': 2e-9,
        'fall_time': 2e-9,
        'pulse_width': 4e-9,
        'period': 1e-5
    }
    c, l = [None for _ in range(21)], [None for _ in range(20)]
    n = lg.create_node(SSrcNode, params=params)
    e = lg.create_edge(LadEdge, w_src=np.random.normal(1, 0.1), w_dst=np.random.normal(1, 0.1))
    c[0] = lg.create_node(CNode, val=1e-9)
    lg.connect(e, src=n, dst=c[0])
    for i in range(10):
        l[i] = lg.create_node(LNode, val=1e-9)
        c[i+1] = lg.create_node(CNode, val=1e-9)
        e1 = lg.create_edge(LadEdge, w_src=np.random.normal(1, 0.1), w_dst=np.random.normal(1, 0.1))
        e2 = lg.create_edge(LadEdge, w_src=np.random.normal(1, 0.1), w_dst=np.random.normal(1, 0.1))
        lg.connect(e1, src=c[i], dst=l[i])
        lg.connect(e2, src=l[i], dst=c[i+1])

    i = 10
    l[i] = lg.create_node(LNode, val=1e-9)
    c[i+1] = lg.create_node(CNode, val=1e-9)
    e1 = lg.create_edge(SwitchLadEdge, w_src=np.random.normal(1, 0.1), w_dst=np.random.normal(1, 0.1), is_on=True)
    e2 = lg.create_edge(LadEdge, w_src=np.random.normal(1, 0.1), w_dst=np.random.normal(1, 0.1))
    lg.connect(e1, src=c[i], dst=l[i])
    lg.connect(e2, src=l[i], dst=c[i+1])

    for i in range(11, 20):
        l[i] = lg.create_node(LNode, val=1e-9)
        c[i+1] = lg.create_node(CNode, val=1e-9)
        e1 = lg.create_edge(LadEdge, w_src=np.random.normal(1, 0.1), w_dst=np.random.normal(1, 0.1))
        e2 = lg.create_edge(LadEdge, w_src=np.random.normal(1, 0.1), w_dst=np.random.normal(1, 0.1))
        lg.connect(e1, src=c[i], dst=l[i])
        lg.connect(e2, src=l[i], dst=c[i+1])

    return lg

def wire_ckt(lg):
    circuit = Circuit('ladder')
    
    circuit.raw_spice += lg.to_spice()
    return circuit

lg = create_graph()    
circuit = wire_ckt(lg)
print(lg.to_dynamical_system())
exec(lg.to_dynamical_system())
analysis = sim(circuit)
render_transient(analysis, 'test.png', {'c_0', 'l_0'})

time_range = [0, 75e-9]
time_points = np.linspace(*time_range, 1000)
states = [0 for _ in range(41)]
sol = solve_ivp(dynamics, [0, 75e-9], states, dense_output=True, max_step=1e-10)
for idx, label in [(21, 'l_0'), (0, 'c_0')]:
    plt.plot(time_points, sol.sol(time_points)[idx].T, label=label)
plt.legend()
plt.savefig('test2.png')