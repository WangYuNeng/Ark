from math import sin
import matplotlib.pyplot as plt
import numpy as np
from saa.ladder_spec import *
from saa.simulator import Simulator

r_val = 1
lc_val = 1e-9
sum_c_val = lc_val
w_src, w_dst = 1, 1
params = {
    'amplitude': 1,
    'delay': 0,
    'rise_time': 5e-9,
    'fall_time': 5e-9,
    'pulse_width': 10e-9,
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
    connect_lad(lg=lg, src=cs_lad2[0], dst=r)
    sws = []
    for i in range(n_switch):
        c_idx = i * n_unit_ladder
        c_src, c_dst = cs_lad1[c_idx], cs_lad2[c_idx]
        sws.append(connect_switchbuf(lg=lg, src=c_src, dst=c_dst))

    return cs_lad1[0], cs_lad2[0], sws


def create_graph(n_unit_ladder, tgt_delay_ladder):

    lg = LadderGraph()
    ns1, ne1, sw1 = create_tunable_seg(lg, n_unit_ladder=n_unit_ladder, n_switch=10)
    ns2, ne2, sw2 = create_tunable_seg(lg, n_unit_ladder=n_unit_ladder, n_switch=10)
    ns3, ne3, sw3 = create_tunable_seg(lg, n_unit_ladder=n_unit_ladder, n_switch=10)
    on_switch_id = tgt_delay_ladder // n_unit_ladder // 2
    sw1[on_switch_id * 2].is_on = True
    sw2[on_switch_id].is_on = True
    sw3[0].is_on = True
    sum_node = lg.create_node(SumNode, val=sum_c_val)
    connect_buf(lg=lg, src=ne1, dst=sum_node)
    connect_buf(lg=lg, src=ne2, dst=sum_node)
    connect_buf(lg=lg, src=ne3, dst=sum_node)

    s1 = lg.create_node(VsNode, params=params.copy(), r=1)
    s2 = lg.create_node(VsNode, params=params.copy(), r=1)
    s3 = lg.create_node(VsNode, params=params.copy(), r=1)
    connect_lad(lg=lg, src=s1, dst=ns1)
    connect_lad(lg=lg, src=s2, dst=ns2)
    connect_lad(lg=lg, src=s3, dst=ns3)

    return [s1, s2, s3], [ns1, ns2, ns3, ne1, ne2, ne3, sum_node], lg

n_unit_ladder = 2
for tgt_delay_ladder in [4, 8, 12]:
    [s1, s2, s3], measured_node, lg = create_graph(n_unit_ladder=n_unit_ladder, tgt_delay_ladder=tgt_delay_ladder)
    c, td, d = 3e8, 1e-9 * tgt_delay_ladder, 50e-1
    print(c*td/d , np.arcsin(c*td/d))
    plt.figure(1)
    plt.title('SPICE simulation, Td={:.2f}ns'.format(td*1e9))
    for theta in [0, 0.25, 0.5, 0.75, 1, 1.25]:
        s1.params['delay'], s2.params['delay'], s3.params['delay'] = 0, d*sin(theta)/c, 2*d*sin(theta)/c
        simulator = Simulator(graph=lg)
        analysis = simulator.spice_sim('theata_{}.png'.format(theta), [node.name for node in measured_node])
        time_data = analysis.time.as_ndarray()
        c0, sum_theta = analysis.nodes['c_0'].as_ndarray(), analysis.nodes['sum_0'].as_ndarray()
        if theta == 0:
            plt.plot(time_data, c0, label='Input')
        plt.plot(time_data, sum_theta, label='Output @ p={:.2f}'.format(theta))
    plt.xlabel('time(s)')
    plt.ylabel('Amplitude(V)')
    plt.legend()
    plt.savefig('spice-{:.2e}.pdf'.format(td))
    plt.clf()


    plt.figure(1)
    plt.title('Dynamical system simulation, Td={:.2f}ns'.format(td*1e9))
    for theta in [0, 0.25, 0.5, 0.75, 1, 1.25]:
        s1.params['delay'], s2.params['delay'], s3.params['delay'] = 0, d*sin(theta)/c, 2*d*sin(theta)/c
        simulator = Simulator(graph=lg)
        time_points, var_to_idx, sol = simulator.ds_sim('theata_{}_ds.png'.format(theta), [node.name for node in measured_node])
        c0, sum_theta = sol.sol(time_points)[var_to_idx['c_0']], sol.sol(time_points)[var_to_idx['sum_0']]
        if theta == 0:
            plt.plot(time_points, c0, label='Input')
        plt.plot(time_points, sum_theta, label='Output @ p={:.2f}'.format(theta))
    
    plt.xlabel('time(s)')
    plt.ylabel('Amplitude(V)')
    plt.legend()
    plt.savefig('ds-{:.2e}.pdf'.format(td))
    plt.clf()
