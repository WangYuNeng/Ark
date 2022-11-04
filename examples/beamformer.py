from math import sin
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


def create_seg(lg: LadderGraph, n_ladder):
    nc0 = lg.create_node(CNode, val=lc_val)
    n_prev = nc0
    for i in range(n_ladder):
        nl = lg.create_node(LNode, val=lc_val)
        nc = lg.create_node(CNode, val=lc_val)
        e1 = lg.create_edge(LadEdge, w_src=w_src, w_dst=w_dst)
        e2 = lg.create_edge(LadEdge, w_src=w_src, w_dst=w_dst)
        lg.connect(edge=e1, src=n_prev, dst=nl)
        lg.connect(edge=e2, src=nl, dst=nc)
        n_prev = nc
    r1 = lg.create_node(RNode, val=r_val)
    r2 = lg.create_node(RNode, val=r_val)
    e1 = lg.create_edge(LadEdge, w_src=w_src, w_dst=w_dst)
    e2 = lg.create_edge(LadEdge, w_src=w_src, w_dst=w_dst)
    lg.connect(edge=e1, src=nc0, dst=r1)
    lg.connect(edge=e2, src=n_prev, dst=r2)
    return nc0, n_prev

def create_graph(ladder_per_seg):

    lg = LadderGraph()
    ns1, ne1 = create_seg(lg, ladder_per_seg * 3)
    ns2, ne2 = create_seg(lg, ladder_per_seg * 2)
    ns3, ne3 = create_seg(lg, ladder_per_seg * 1)
    sum_node = lg.create_node(SumNode, val=sum_c_val)
    e1 = lg.create_edge(LadEdge, w_src=0, w_dst=w_dst)
    e2 = lg.create_edge(LadEdge, w_src=0, w_dst=w_dst)
    e3 = lg.create_edge(LadEdge, w_src=0, w_dst=w_dst)
    lg.connect(edge=e1, src=ne1, dst=sum_node)
    lg.connect(edge=e2, src=ne2, dst=sum_node)
    lg.connect(edge=e3, src=ne3, dst=sum_node)

    s1 = lg.create_node(VsNode, params=params.copy(), r=1)
    s2 = lg.create_node(VsNode, params=params.copy(), r=1)
    s3 = lg.create_node(VsNode, params=params.copy(), r=1)
    e1 = lg.create_edge(LadEdge, w_src=0, w_dst=w_dst)
    e2 = lg.create_edge(LadEdge, w_src=0, w_dst=w_dst)
    e3 = lg.create_edge(LadEdge, w_src=0, w_dst=w_dst)
    lg.connect(edge=e1, src=s1, dst=ns1)
    lg.connect(edge=e2, src=s2, dst=ns2)
    lg.connect(edge=e3, src=s3, dst=ns3)

    return [s1, s2, s3], [ns1, ns2, ns3, ne1, ne2, ne3, sum_node], lg

ladder_per_seg = 10
[s1, s2, s3], measured_node, lg = create_graph(ladder_per_seg=ladder_per_seg)
c, td, d = 3e8, 1e-9 * ladder_per_seg, 50e-1
print(c*td/d , np.arcsin(c*td/d))
for theta in np.arange(0, np.pi/2, 0.1):
    s1.params['delay'], s2.params['delay'], s3.params['delay'] = 0, d*sin(theta)/c, 2*d*sin(theta)/c
    simulator = Simulator(graph=lg)
    simulator.spice_sim('theata_{}.png'.format(theta), [node.name for node in measured_node])
    simulator.ds_sim('theata_{}_ds.png'.format(theta), [node.name for node in measured_node])