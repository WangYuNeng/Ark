import numpy as np
import matplotlib.pyplot as plt
from math import sin
from scipy.integrate import solve_ivp
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit

from saa.graph import Graph

class Simulator:

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def spice_sim(self, file_name, data_points, temperature=25, nominal_temperature=25, step_time=1e-10, end_time=75e-9):
        circuit = Circuit('simulation')
        # print('spice_sim')
        # print(self._graph.to_spice())
        circuit.raw_spice += self._graph.to_spice()
        simulator = circuit.simulator(temperature=temperature, nominal_temperature=nominal_temperature)
        analysis = simulator.transient(step_time=step_time, end_time=end_time)

        # time_data = analysis.time.as_ndarray()
        # print(self._graph.to_spice())
        # plt.figure(1)
        # for name in data_points:
        #     data = analysis.nodes[name]
        #     series_name = name
        #     series_data = data.as_ndarray()
        #     plt.plot(time_data,series_data, label=series_name)
        # plt.legend()
        # plt.savefig(file_name)
        # plt.clf()
        return analysis
        return time_data, analysis.nodes['c_0'].as_ndarray(), analysis.nodes['sum_0'].as_ndarray()

    def ds_sim(self, file_name, data_points, time_range=[0, 75e-9], n_time_points=1000):
        var_to_idx, ds_str = self._graph.to_dynamical_system()
        # print('ds_sim')
        # print(ds_str)
        exec(ds_str, globals(), globals())
        n_states = len(var_to_idx)
        time_points = np.linspace(*time_range, n_time_points)
        states = [0 for _ in range(n_states)]
        sol = solve_ivp(dynamics, time_range, states, dense_output=True, max_step=1e-10)
        # plt.figure(1)
        # for name in data_points:
        #     idx = var_to_idx[name]
        #     plt.plot(time_points, sol.sol(time_points)[idx].T, label=name)
        # plt.legend()
        # plt.savefig(file_name)
        # plt.clf()
        return time_points, var_to_idx, sol
        return time_points, sol.sol(time_points)[var_to_idx['c_0']], sol.sol(time_points)[var_to_idx['sum_0']]