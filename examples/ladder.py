import numpy as np
from saa.ladder_spec import *
from saa.simulator import Simulator
# np.random.seed(0)

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
    n = lg.create_node(VsNode, params=params, r=1)
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

lg = create_graph() 
simulator = Simulator(graph=lg)
simulator.spice_sim('test.png', ['c_0', 'c_20'])
simulator.ds_sim('test2.png', ['c_0', 'c_20'])