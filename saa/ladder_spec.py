from operator import is_
from saa.node import Node, StatefulNode, StatelessNode
from saa.edge import Edge

class LadEdge(Edge):

    def __init__(self, id, w_src, w_dst) -> None:
        super().__init__(f'le_{id}')
        self._w_src, self._w_dst = w_src, w_dst

    @property
    def w_src(self):
        return self._w_src

    @property
    def w_dst(self):
        return self._w_dst

    def validation(self) -> bool:
        raise NotImplementedError

    def to_dynamical_system(self) -> str:
        return None

    def to_spice(self) -> str:
        return None

class SwitchLadEdge(Edge):

    def __init__(self, id, w_src, w_dst, is_on) -> None:
        super().__init__(f'se_{id}')
        self._w_src, self._w_dst = w_src, w_dst
        self._is_on = is_on

    @property
    def is_on(self):
        return self._is_on

    @property
    def w_src(self):
        if self._is_on:
            return self._w_src
        return 0

    @property
    def w_dst(self):
        if self._is_on:
            return self._w_dst
        return 0

    def linked_spice_name(self, n) -> str:
        if n == self.src:
            return self._switched_name(self.dst.name)
        elif n == self.dst:
            return self._switched_name(self.src.name)
        else:
            assert False, 'edge {} does not connect to node {}'.format(self.name, n.name)

    def validation(self) -> bool:
        raise NotImplementedError

    def to_dynamical_system(self) -> str:
        return None

    def to_spice(self) -> str:
        volt = self.is_on * 2 - 1
        ctrlp, ctrln = '{}p'.format(self.name), '0'
        ctrl_str = 'V{} {} {} {}'.format(self.name, ctrlp, ctrln, volt)
        sw1_str = 'X{}_src {} {} {} {} switch'.format(self.name, self.src.name, 
            self._switched_name(self.src.name), ctrlp, ctrln)
        sw2_str = 'X{}_dst {} {} {} {} switch'.format(self.name, self.dst.name, 
            self._switched_name(self.dst.name), ctrlp, ctrln)
        return '\n'.join([ctrl_str, sw1_str, sw2_str])

    def _switched_name(self, name):
        return '{}_{}_swo'.format(self.name, name)


class RNode(StatelessNode):

    def __init__(self, id, val) -> None:
        super().__init__(f'r_{id}')
        self._r = val

    @property
    def r(self):
        return self._r

    def validation(self) -> bool:
        raise NotImplementedError

    def to_dynamical_system(self) -> str:
        return None

    def to_spice(self) -> str:
        return None

class GNode(StatelessNode):

    def __init__(self) -> None:
        super().__init__('0')

    def validation(self) -> bool:
        raise NotImplementedError

    def to_dynamical_system(self) -> str:
        raise NotImplementedError

    def to_spice(self) -> str:
        raise NotImplementedError

class SSrcNode(StatelessNode):
    '''
    Pulse source for now
    '''

    def __init__(self, id, params) -> None:
        super().__init__(f'ssrc_{id}')
        self.params = params

    def validation(self) -> bool:
        raise NotImplementedError

    def to_dynamical_system(self) -> str:
        return '{} = pulse(t, amplitude={}, delay={}, rise_time={}, fall_time={}, pulse_width={}, period={})'.format(
            self.name, self.params['amplitude'], self.params['delay'], self.params['rise_time'],
            self.params['fall_time'], self.params['pulse_width'], self.params['period']
        )

    def to_spice(self) -> str:
        return 'V{} {} 0 DC 0V PULSE(0V {}V {}s {}s {}s {}s {}s)'.format(
            self.name, self.name, self.params['amplitude'], self.params['delay'], self.params['rise_time'],
            self.params['fall_time'], self.params['pulse_width'], self.params['period']
        )

class LCNode(StatefulNode):

    gm_factor = 1e-3
    Rloss = 1e12
    model = 'gmc'

    def __init__(self, node_type, id, val) -> None:
        super().__init__(f'{node_type}_{id}')
        self._val = val

    def validation(self) -> bool:
        raise NotImplementedError

    def to_dynamical_system(self) -> str:
        
        rhs = []
        for edge in self.conn:
            assert not isinstance(edge.src, RNode), 'RNode cannot be src'
            if isinstance(edge.dst, RNode):
                rhs_str = '(-{:.3e}/{:.3e} * {})'.format(edge.w_src, edge.dst.r, self.name)
            elif self == edge.src:
                rhs_str = '(-{:.3e} * {})'.format(edge.w_src, edge.linked_name(self))
            elif self == edge.dst:
                rhs_str = '({:.3e} * {})'.format(edge.w_dst, edge.linked_name(self))
            else:
                assert False, 'edge {} does not connect to node {}'.format(edge.name, self.name)
            rhs.append(rhs_str)
        ddt_str = '{} = 1/{:.3e} * '.format(self.ddt_name, self._val) + '({})'.format(' + '.join(rhs))
        return ddt_str

    def to_spice(self) -> str:
        component = 'X{}'.format(self.name)
        ins, gms = [], []
        for edge in self.conn:
            assert not isinstance(edge.src, RNode), 'RNode cannot be src'
            if isinstance(edge.dst, RNode):
                in_name, in_gm = self.name, -edge.w_src * self.gm_factor
            elif self == edge.src:
                in_name, in_gm = edge.linked_spice_name(self), -edge.w_src * self.gm_factor
            elif self == edge.dst:
                in_name, in_gm = edge.linked_spice_name(self), edge.w_dst * self.gm_factor
            else:
                assert False, 'edge {} does not connect to node {}'.format(edge.name, self.name)
            ins.append(in_name)
            gms.append(in_gm)

        component = 'X{}'.format(self.name)
        params = 'Cint={:.3e} Rloss={:.3e} '.format(self._val * self.gm_factor, self.Rloss) + \
            ' '.join(['gm{}={:.3e}'.format(i, gm) for i, gm in enumerate(gms)])
        inputs = ' '.join(ins)
        output = self.name
        model = '{}{}'.format(self.model, len(ins))
        spice_str = ' '.join([component, inputs, output, model, params])
        return spice_str

class CNode(LCNode):

    def __init__(self, id, val) -> None:
        super().__init__('c', id, val)

class LNode(LCNode):

    def __init__(self, id, val) -> None:
        super().__init__('l', id, val)

class SumNode(LCNode):

    Rloss = 1e3

    def __init__(self, id, val) -> None:
        super().__init__('sum', id, val)

class LadderGraph:

    def __init__(self) -> None:
        self.sl_node_types = [RNode, SSrcNode]
        self.sf_node_types = [CNode, LNode, SumNode]
        self.edge_types = [LadEdge, SwitchLadEdge]
        self.node_type_dict = {nt: [] for nt in self.sl_node_types + self.sf_node_types}
        self.edge_type_dict = {et: [] for et in self.edge_types}

    def create_node(self, node_type, **kwargs):
        assert node_type in self.node_type_dict
        nodes = self.node_type_dict[node_type]
        node = node_type(len(nodes), **kwargs)
        nodes.append(node)
        return node

    def create_edge(self, edge_type, **kwargs):
        assert edge_type in self.edge_type_dict
        edges = self.edge_type_dict[edge_type]
        edge = edge_type(len(edges), **kwargs)
        edges.append(edge)
        return edge

    def connect(self, edge: Edge, src: Node, dst: Node):
        edge.connect(src=src, dst=dst)
        src.add_conn(edge=edge)
        dst.add_conn(edge=edge)

    def to_dynamical_system(self):
        ds_strs, unpack_var, rtv = [], [], []

        for et in self.edge_types:
            for edge in self.edge_type_dict[et]:
                ds_str = edge.to_dynamical_system()
                if ds_str:
                    ds_strs.append(ds_str)

        for nt in self.sl_node_types:
            for node in self.node_type_dict[nt]:
                ds_str = node.to_dynamical_system()
                if ds_str:
                    ds_strs.append(ds_str)

        for nt in self.sf_node_types:
            for node in self.node_type_dict[nt]:
                ds_str = node.to_dynamical_system()
                ds_strs.append(ds_str)
                unpack_var.append(node.name)
                rtv.append(node.ddt_name)
        fn_def = 'def dynamics(t, x):'
        unpack = ', '.join(unpack_var) + ' = x'
        rt = 'return [{}]'.format(', '.join(rtv))
        return '\n\t'.join([fn_def, unpack] + ds_strs + [rt])

    def to_spice(self):
        spice_strs = self._base_ckt()

        for nt in self.sl_node_types:
            for node in self.node_type_dict[nt]:
                spice_str = node.to_spice()
                if spice_str:
                    spice_strs.append(spice_str)

        for et in self.edge_types:
            for edge in self.edge_type_dict[et]:
                spice_str = edge.to_spice()
                if spice_str:
                    spice_strs.append(spice_str)
        
        for nt in self.sf_node_types:
            for node in self.node_type_dict[nt]:
                spice_str = node.to_spice()
                if spice_str:
                    spice_strs.append(spice_str)
        return '\n'.join(spice_strs)

    def _base_ckt(self):
        sub_strs = []
        for n_in in range(1,4):
            vi_str = ' '.join(['vi{}'.format(i) for i in range(n_in)])
            gm_str = ' '.join(['gm{}=1e-3'.format(i) for i in range(n_in)])
            sub_strs.append('.subckt gmc{} {} vo Cint=1e-12 {} Rloss=1e12'.format(n_in, vi_str, gm_str))
            sub_strs += ['Gvccs{} 0 vo vi{} 0 gm{}'.format(i, i, i) for i in range(n_in)]
            sub_strs.append('Rr vo 0 r=Rloss')
            sub_strs.append('Cc vo 0 c=Cint')
            sub_strs.append('.ends gmc{}'.format(n_in))
        
        sub_strs.append('.model switch_ sw vt=0 vh =0 ron=1 roff=1e20')
        sub_strs.append('.subckt switch vi vo vctrlp vctrln')
        sub_strs.append('SS1 vi vo vctrlp vctrln switch_ off')
        sub_strs.append('SS2 vo 0 vctrln vctrlp switch_ off')
        sub_strs.append('.ends switch')
        return sub_strs

# lg = LadderGraph()
# params = {
#     'amplitude': 1,
#     'delay': 0,
#     'rise_time': 10e-9,
#     'fall_time': 10e-9,
#     'pulse_width': 20e-9,
#     'period': 1e-5
# }
# n1 = lg.create_node(LNode, val=1e-9)
# n2 = lg.create_node(CNode, val=1e-9)
# n3 = lg.create_node(SSrcNode, params=params)
# e1 = lg.create_edge(LadEdge, w_src=1, w_dst=1)
# e2 = lg.create_edge(LadEdge, w_src=1, w_dst=1)
# lg.connect(e1, src=n1, dst=n2)
# lg.connect(e2, src=n3, dst=n2)
# print(lg.to_dynamical_system())
# print(lg.to_spice())