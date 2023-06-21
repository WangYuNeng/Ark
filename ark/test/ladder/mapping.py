import re
from ark.test.ladder.model import LadderModel
from ark.cdg.cdg import CDG, CDGNode, CDGEdge

GM_FACTOR = 1e-3
MODEL = LadderModel()

FUNC_PATTERN = r'([\w|_]+)=([0-9|e|\.|\+|\-]+)'
RE_OBJ = re.compile(FUNC_PATTERN)

class SpiceMapper:

    def __init__(self) -> None:
        pass

    def to_spice(self, graph: CDG):

        node: CDGNode
        spice_strs: list

        self._gmc_ins = set()
        self._gmrc_ins = set()

        self._graph = graph
        spice_strs = []

        for node in self._graph.nodes:
            if node.cdg_type == MODEL.VN or node.cdg_type == MODEL.IN:
                spice_strs.append(self._map_LC(node=node))

            elif node.cdg_type == MODEL.S:
                spice_strs.append(self._map_src(node=node))

        
        for edge in self._graph.edges:
            pass
        spice_strs += self._base_ckt()

        return '\n'.join(spice_strs)

    
    def _map_LC(self, node: CDGNode):


        def calc_rloss():
            edge: CDGEdge
            
            re_rloss = []
            for edge in node.edges:

                if edge.dst.cdg_type == MODEL.R:
                    inv_r = float(edge.dst.attrs['r'])
                elif edge.src.cdg_type == MODEL.S:
                    inv_r = float(edge.src.attrs['r'])
                else:
                    continue
                
                if node.cdg_type == MODEL.VN:
                    inv_r = 1 / inv_r
                re_rloss.append(inv_r)
            if re_rloss == []:
                return None
            return 1/sum(re_rloss)/GM_FACTOR


        edge: CDGEdge
        component = 'X{}'.format(node.name)
        ins, gms = list(), list()
        for edge in node.edges:

            if node.get_neighbor(edge=edge).cdg_type == MODEL.R:
                continue

            if node.is_src(edge=edge):
                in_name, in_gm = edge.dst.name, -float(edge.attrs['q_src']) * GM_FACTOR
            else:
                in_name, in_gm = edge.src.name, float(edge.attrs['q_dst']) * GM_FACTOR

            ins.append(in_name)
            gms.append(in_gm)
        
        component = 'X{}'.format(node.name)

        if node.cdg_type == MODEL.VN:
            base_val = float(node.attrs['c'])
        elif node.cdg_type == MODEL.IN:
            base_val = float(node.attrs['l'])
    
        rloss = calc_rloss()
        if rloss == None:
            model_prefix = 'gmc'
            self._gmc_ins.add(len(ins))
            params = 'Cint={} '.format(base_val * GM_FACTOR) + \
                ' '.join(['gm{}={}'.format(i, gm) for i, gm in enumerate(gms)])
        else:
            model_prefix = 'gmrc'
            self._gmrc_ins.add(len(ins))
            params = 'Cint={} Rloss={} '.format(base_val * GM_FACTOR, rloss) + \
                ' '.join(['gm{}={}'.format(i, gm) for i, gm in enumerate(gms)])

        inputs = ' '.join(ins)
        output = node.name
        model = '{}{}'.format(model_prefix, len(ins))
        spice_str = ' '.join([component, inputs, output, model, params])
        return spice_str

    def _map_src(self, node: CDGNode):

        def parse_attr(attr_str):

            matches = RE_OBJ.findall(attr_str)
            params = {m[0]: float(m[1]) for m in matches}
            return params['amplitude'], params['delay'], params['rise_time'], \
                params['fall_time'], params['pulse_width'], params['period']
        
        amplitude, delay, rise_time, fall_time, pulse_width, period = parse_attr(node.attrs['fn'])

        if node.edges[0].dst.cdg_type == MODEL.VN:
            amplitude /= float(node.attrs['r'])
        return 'V{} {} 0 DC 0V PULSE(0V {}V {}s {}s {}s {}s {}s)'.format(
            node.name, node.name, amplitude, delay, rise_time, fall_time, pulse_width, period
        )


    def _base_ckt(self):
        sub_strs = []
        for n_in in self._gmrc_ins:
            vi_str = ' '.join(['vi{}'.format(i) for i in range(n_in)])
            gm_str = ' '.join(['gm{}=1e-3'.format(i) for i in range(n_in)])
            sub_strs.append('.subckt gmrc{} {} vo Cint=1e-12 {} Rloss=1e12'.format(n_in, vi_str, gm_str))
            sub_strs += ['Gvccs{} 0 vo vi{} 0 gm{}'.format(i, i, i) for i in range(n_in)]
            sub_strs.append('Rr vo 0 r=Rloss')
            sub_strs.append('Cc vo 0 c=Cint')
            sub_strs.append('.ends gmrc{}'.format(n_in))
        for n_in in self._gmc_ins:
            vi_str = ' '.join(['vi{}'.format(i) for i in range(n_in)])
            gm_str = ' '.join(['gm{}=1e-3'.format(i) for i in range(n_in)])
            sub_strs.append('.subckt gmc{} {} vo Cint=1e-12 {}'.format(n_in, vi_str, gm_str))
            sub_strs += ['Gvccs{} 0 vo vi{} 0 gm{}'.format(i, i, i) for i in range(n_in)]
            sub_strs.append('Cc vo 0 c=Cint')
            sub_strs.append('.ends gmc{}'.format(n_in))

        return sub_strs