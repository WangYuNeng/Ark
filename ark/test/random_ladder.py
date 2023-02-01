import numpy as np
from ark.test.ladder.model import LadderModel
from ark.cdg.cdg import CDG

class Sampler:

    def __init__(self, seed) -> None:
        np.random.seed(seed=seed)

    def sample_float(self, low, high):
        return np.random.uniform(low=low, high=high)
    
    def sample_int(self, low, high):
        return np.random.randint(low=low, high=high)

    def sample_choice(self, items, prob=None):
        return np.random.choice(a=items, p=prob)


class Generator:

    def __init__(self) -> None:
        self._ladder_model = LadderModel()
        self._sampler = None

        m = self._ladder_model
        self._type2elements = {m.VN: [], m.IN: [], m.S: [], m.R: [], m.E: []}
        self._type2prefix = {m.VN: 'c', m.IN: 'l', m.S: 's', m.R: 'r', m.E: 'e'}

    def generate(self, seed) -> CDG:
        self._sampler = Sampler(seed=seed)
        graph = CDG()
        self._initialize(graph=graph)

        return graph


    def gen_node(self, graph: CDG, cdg_type):

        attrs = self._gen_attrs(cdg_type=cdg_type)

        if cdg_type == self._ladder_model.S:
            attrs = {'fn': self._pulse_fn_str(attrs=attrs), 'r': attrs['r']}

        nodes = self._type2elements[cdg_type]
        name = self._type_id_str(cdg_type, len(nodes))
        
        node = graph.add_node(name=name, cdg_type=cdg_type, attrs=attrs)
        nodes.append(node)
        return node

    
    def gen_edge(self, graph: CDG, cdg_type, src, dst):
        attrs = self._gen_attrs(cdg_type=cdg_type)

        edges = self._type2elements[cdg_type]
        name = self._type_id_str(cdg_type, len(edges))
        
        edge= graph.add_edge(name=name, cdg_type=cdg_type, attrs=attrs, src=src, dst=dst)
        edges.append(edge)
        return edge

    def gen_source(self, graph):
        return self.gen_node(graph=graph, cdg_type=self._ladder_model.S)
    
    def gen_vn(self, graph):
        return self.gen_node(graph=graph, cdg_type=self._ladder_model.VN)
    
    def gen_in(self, graph):
        return self.gen_node(graph=graph, cdg_type=self._ladder_model.IN)
    
    def gen_r(self, graph):
        return self.gen_node(graph=graph, cdg_type=self._ladder_model.R)

    def _gen_attrs(self, cdg_type):

        param_range = self._ladder_model.get_param_range(cdg_type=cdg_type)
        attrs = {key: self._sampler.sample_float(val[0], val[1]) for key, val in param_range.items()}

        return attrs

    def _initialize(self, graph):
        m = self._ladder_model
        s = self._sampler

        source = self.gen_source(graph=graph)
        sampled_type = s.sample_choice([m.IN, m.VN], [0.5, 0.5])
        node = self.gen_node(graph=graph, cdg_type=sampled_type)
        edge = self.gen_edge(graph=graph, cdg_type=m.E, src=source, dst=node)

    def _type_id_str(self, t, i):
        return '{}{}'.format(self._type2prefix[t], i)

    def _pulse_fn_str(self, attrs):
        return 'pulse(t, amplitude={}, delay={}, rise_time={}, fall_time={}, pulse_width={}, period={}'.format(
            attrs['amplitude'], attrs['delay'], attrs['rise_time'], attrs['fall_time'], 
            attrs['pulse_width'], attrs['period']
        )
        
    
if __name__ == '__main__':
    g = Generator()
    from ark.test.ladder.mapping import SpiceMapper

    mapper = SpiceMapper()
    print(mapper.to_spice(g.generate(428)))