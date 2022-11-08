import matplotlib.pyplot as plt
import numpy as np
from saa.ladder_spec import *
from saa.simulator import Simulator

r_val = 1
lc_val = 1e-9
sum_c_val = lc_val
w_src, w_dst = 1, 1
params1 = {
    'amplitude': 1,
    'delay': 0,
    'rise_time': 5e-9,
    'fall_time': 5e-9,
    'pulse_width': 10e-9,
    'period': 1e-5
}
params2 = {
    'amplitude': 0.9,
    'delay': 0.3e-9,
    'rise_time': 8e-9,
    'fall_time': 6e-9,
    'pulse_width': 8e-9,
    'period': 1e-5
}
params3 = {
    'amplitude': 1.2,
    'delay': 1e-9,
    'rise_time': 6e-9,
    'fall_time': 7e-9,
    'pulse_width': 12e-9,
    'period': 1e-5
}
params4 = {
    'amplitude': 0.75,
    'delay': 3e-9,
    'rise_time': 7e-9,
    'fall_time': 10e-9,
    'pulse_width': 4e-9,
    'period': 1e-5
}


# lazy implementation of connection, fix in the future
def connect_lad(lg: LadderGraph, src, dst):
    e = lg.create_edge(LadEdge, w_src=w_src, w_dst=w_dst)
    lg.connect(edge=e, src=src, dst=dst)
    return e

def connect_buf(lg: LadderGraph, src, dst):
    e = lg.create_edge(LadEdge, w_src=0, w_dst=w_dst)
    lg.connect(edge=e, src=src, dst=dst)
    return e

def connect_bufn(lg: LadderGraph, src, dst):
    e = lg.create_edge(LadEdge, w_src=0, w_dst=-w_dst)
    lg.connect(edge=e, src=src, dst=dst)
    return e

def connect_switchbuf(lg: LadderGraph, src, dst):
    e = lg.create_edge(SwitchLadEdge, w_src=0, w_dst=w_dst, is_on=False)
    lg.connect(edge=e, src=src, dst=dst)
    return e
# lazy implementation of connection, fix in the future

def create_seg(lg: LadderGraph, n_ladder):
    ncs = []
    ncs.append(lg.create_node(CNode, val=lc_val))
    for i in range(n_ladder):
        nl = lg.create_node(LNode, val=lc_val)
        nc = lg.create_node(CNode, val=lc_val)
        connect_lad(lg=lg, src=ncs[-1], dst=nl)
        connect_lad(lg=lg, src=nl, dst=nc)
        ncs.append(nc)
    r = lg.create_node(RNode, val=r_val)
    connect_lad(lg=lg, src=ncs[-1], dst=r)
    return ncs

def create_tunable_seg(lg: LadderGraph, n_unit_ladder, n_switch):
    n_ladder = n_unit_ladder * (n_switch - 1)
    cs_lad1 = create_seg(lg=lg, n_ladder=n_ladder)
    cs_lad2 = create_seg(lg=lg, n_ladder=n_ladder)
    r = lg.create_node(RNode, val=r_val)
    r2 = lg.create_node(RNode, val=r_val)
    connect_lad(lg=lg, src=cs_lad1[0], dst=r2)
    connect_lad(lg=lg, src=cs_lad2[0], dst=r)
    sws = []
    for i in range(n_switch):
        c_idx = i * n_unit_ladder
        c_src, c_dst = cs_lad1[c_idx], cs_lad2[c_idx]
        sws.append(connect_switchbuf(lg=lg, src=c_src, dst=c_dst))

    return cs_lad1[0], cs_lad2[0], sws


def create_graph(n_unit_ladder, n_self_delay_ladder):

    lg = LadderGraph()
    ns1, ne1, sw1 = create_tunable_seg(lg, n_unit_ladder=n_unit_ladder, n_switch=10)
    ns2, ne2, sw2 = create_tunable_seg(lg, n_unit_ladder=n_unit_ladder, n_switch=10)
    
    sw1[n_self_delay_ladder].is_on = True
    sw2[0].is_on = True
    diff_node = lg.create_node(SumNode, val=sum_c_val)
    connect_bufn(lg=lg, src=ne1, dst=diff_node)
    connect_buf(lg=lg, src=ne2, dst=diff_node)

    sum_node = lg.create_node(SumNode, val=sum_c_val)
    s1 = lg.create_node(VsNode, params=params1, r=1)
    s2 = lg.create_node(VsNode, params=params2, r=1)
    s3 = lg.create_node(VsNode, params=params3, r=1)
    s4 = lg.create_node(VsNode, params=params4, r=1)
    connect_buf(lg=lg, src=s1, dst=sum_node)
    connect_buf(lg=lg, src=s2, dst=sum_node)
    connect_buf(lg=lg, src=s3, dst=sum_node)
    connect_buf(lg=lg, src=s4, dst=sum_node)
    connect_buf(lg=lg, src=sum_node, dst=ns1)
    connect_buf(lg=lg, src=sum_node, dst=ns2)

    return [s1, s2, s3, s4], [diff_node], lg

plt.figure(1)
for n_self_delay_ladder in [1,2,4]:
    n_unit_ladder = 4
    [s1, s2, s3, s4], measured_node, lg = create_graph(n_unit_ladder=n_unit_ladder, n_self_delay_ladder=n_self_delay_ladder)
    simulator = Simulator(graph=lg)
    analysis = simulator.spice_sim('sense/spice.png', [node.name for node in measured_node])
    time_data = analysis.time.as_ndarray()
    data_points = [node.name for node in measured_node]
    if n_self_delay_ladder == 1:
        for i in range(4):
            name = 'ssrc_{}'.format(i)
            data = analysis.nodes[name]
            series_name = name
            series_data = data.as_ndarray()
            plt.plot(time_data,series_data, label='Sensor value {}'.format(i))
    for name in data_points:
        data = analysis.nodes[name]
        series_name = name
        series_data = data.as_ndarray()
        plt.plot(time_data,series_data, label='Output @ d={}ns'.format(n_self_delay_ladder*n_unit_ladder*2))

plt.xlabel('time(s)')
plt.ylabel('Amplitude(V)')
plt.legend()
plt.savefig('sense/spice.pdf'.format(n_self_delay_ladder))
plt.clf()

plt.figure(1)
for n_self_delay_ladder in [1,2,4]:
    n_unit_ladder = 4
    [s1, s2, s3, s4], measured_node, lg = create_graph(n_unit_ladder=n_unit_ladder, n_self_delay_ladder=n_self_delay_ladder)
    simulator = Simulator(graph=lg)
    time_points, var_to_idx, sol = simulator.ds_sim('sense/ds.png', [node.name for node in measured_node])
    if n_self_delay_ladder == 1:
        for i in range(4):
            name = 'ssrc_{}'.format(i)
            data = analysis.nodes[name]
            series_name = name
            series_data = data.as_ndarray()
            plt.plot(time_data,series_data, label='Sensor value {}'.format(i))
    for name in data_points:
        idx = var_to_idx[name]
        plt.plot(time_points, sol.sol(time_points)[idx].T, label='Output @ d={}ns'.format(n_self_delay_ladder*n_unit_ladder*2))
plt.xlabel('time(s)')
plt.ylabel('Amplitude(V)')
plt.legend()
plt.savefig('sense/ds.pdf'.format(n_self_delay_ladder))
plt.clf()