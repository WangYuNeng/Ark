import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit

from ark.test.ladder.model import LadderModel
from ark.cdg.cdg import CDG, CDGNode, CDGType

class Sampler:

    def __init__(self, seed) -> None:
        np.random.seed(seed=seed)

    def sample_float(self, low, high):
        return np.random.uniform(low=low, high=high)
    
    def sample_int(self, low, high):
        return np.random.randint(low=low, high=high)

    def sample_choice(self, choices, prob=None):
        return np.random.choice(a=choices, p=prob)


class Generator:

    def __init__(self) -> None:
        self._ladder_model = LadderModel()
        self._sampler = None

        m = self._ladder_model
        self._type2elements = {}
        self._type2prefix = {m.VN: 'c', m.IN: 'l', m.S: 's', m.R: 'r', m.E: 'e'}

    def generate(self, max_op, seed) -> CDG:
        self._sampler = Sampler(seed=seed)
        graph = CDG()
        m = self._ladder_model
        s = self._sampler
        self._type2elements = {m.VN: [], m.IN: [], m.S: [], m.R: [], m.E: []}
        self._initialize(graph=graph)

        n_op = self._sampler.sample_int(0, max_op + 1)

        vnodes, inodes = self._type2elements[m.VN], self._type2elements[m.IN]
        for _ in range(n_op):
            choices = vnodes + inodes
            node = s.sample_choice(choices=choices)
            while not self._produce(graph, node):
                node = s.sample_choice(choices=choices)

        return graph

    def _produce(self, graph: CDG, node: CDGNode):

        m = self._ladder_model
        s = self._sampler

        def new_source(graph: CDG, node: CDGNode):
            src = self._gen_S(graph=graph)
            edge = self._gen_edge(graph=graph, cdg_type=m.E, src=src, dst=node)
            return True

        def new_r(graph: CDG, node: CDGNode):
            res = self._gen_R(graph=graph)
            edge = self._gen_edge(graph=graph, cdg_type=m.E, src=node, dst=res)
            return True

        def new_vn_or_in(graph: CDG, node: CDGNode):
            if node.cdg_type == m.VN:
                n_node = self._gen_IN(graph=graph)
            elif node.cdg_type == m.IN:
                n_node = self._gen_VN(graph=graph)
            else:
                assert False, 'Only generate VN or IN node'
            return n_node

        def new_out(graph: CDG, node: CDGNode):
            n_node = new_vn_or_in(graph=graph, node=node)
            edge = self._gen_edge(graph=graph, cdg_type=m.E, src=node, dst=n_node)
            return True

        def new_in(graph: CDG, node: CDGNode):
            n_node = new_vn_or_in(graph=graph, node=node)
            edge = self._gen_edge(graph=graph, cdg_type=m.E, src=n_node, dst=node)
            return True

        def sample_valid_VN(node: CDGNode):
            if self._type2elements[m.VN] == []:
                return False
            vnode = s.sample_choice(self._type2elements[m.VN])
            if node.is_neighbor(vnode):
                return False
            return vnode

        def connect_out(graph: CDG, node: CDGNode):
            vnode = sample_valid_VN(node=node)
            if vnode:
                edge = self._gen_edge(graph=graph, cdg_type=m.E, src=node, dst=vnode)
                return True
            return False

        def connect_in(graph: CDG, node: CDGNode):
            vnode = sample_valid_VN(node=node)
            if vnode:
                edge = self._gen_edge(graph=graph, cdg_type=m.E, src=vnode, dst=node)
                return True
            return False

        def filter_choices(node: CDGNode, choices: set):
            
            this_type = node.cdg_type
            if node.cdg_type == m.VN:
                choices.remove(connect_out)
                choices.remove(connect_in)

            for edge in node.edges:
                
                nb_type = node.get_neighbor(edge=edge).cdg_type

                if nb_type == m.R:
                    choices.remove(new_r)
                    continue

                if this_type == m.IN:
                    if node.is_src(edge=edge):
                        if new_out in choices:
                            choices.remove(new_out)
                        if connect_out in choices:
                            choices.remove(connect_out)
                    if node.is_dst(edge=edge):
                        if new_in in choices:
                            choices.remove(new_in)
                        if connect_in in choices:
                            choices.remove(connect_in)
                if nb_type == m.S:
                    choices.remove(new_source)


            return choices

        choices = filter_choices(node=node, choices={new_source, new_r, new_out, new_in, connect_out, connect_in})
        if len(choices) == 0:
            return False
        choice_func = s.sample_choice(choices=list(choices))
        return choice_func(graph, node)

    def _gen_node(self, graph: CDG, cdg_type: CDGType):

        attrs = self._gen_attrs(cdg_type=cdg_type)

        if cdg_type == self._ladder_model.S:
            attrs = {'fn': self._pulse_fn_str(attrs=attrs), 'r': attrs['r']}

        nodes = self._type2elements[cdg_type]
        name = self._type_id_str(cdg_type, len(nodes))
        
        node = graph.add_node(name=name, cdg_type=cdg_type, attrs=attrs)
        print(node)
        nodes.append(node)
        return node

    
    def _gen_edge(self, graph: CDG, cdg_type: CDGType, src: CDGNode, dst: CDGNode):
        attrs = self._gen_attrs(cdg_type=cdg_type)

        edges = self._type2elements[cdg_type]
        name = self._type_id_str(cdg_type, len(edges))
        
        edge = graph.add_edge(name=name, cdg_type=cdg_type, attrs=attrs, src=src, dst=dst)
        print(edge)
        edges.append(edge)
        return edge

    def _gen_S(self, graph: CDG):
        return self._gen_node(graph=graph, cdg_type=self._ladder_model.S)
    
    def _gen_VN(self, graph: CDG):
        return self._gen_node(graph=graph, cdg_type=self._ladder_model.VN)
    
    def _gen_IN(self, graph: CDG):
        return self._gen_node(graph=graph, cdg_type=self._ladder_model.IN)
    
    def _gen_R(self, graph: CDG):
        return self._gen_node(graph=graph, cdg_type=self._ladder_model.R)

    def _gen_attrs(self, cdg_type: CDGType):

        param_range = self._ladder_model.get_param_range(cdg_type=cdg_type)
        attrs = {key: str(self._sampler.sample_float(val[0], val[1])) for key, val in param_range.items()}

        return attrs

    def _initialize(self, graph: CDG):
        m = self._ladder_model
        s = self._sampler

        source = self._gen_S(graph=graph)
        sampled_type = s.sample_choice([m.IN, m.VN], [0.5, 0.5])
        node = self._gen_node(graph=graph, cdg_type=sampled_type)
        edge = self._gen_edge(graph=graph, cdg_type=m.E, src=source, dst=node)

    def _type_id_str(self, t, i):
        return '{}{}'.format(self._type2prefix[t], i)

    def _pulse_fn_str(self, attrs):
        return 'pulse(t, amplitude={}, delay={}, rise_time={}, fall_time={}, pulse_width={}, period={})'.format(
            attrs['amplitude'], attrs['delay'], attrs['rise_time'], attrs['fall_time'], 
            attrs['pulse_width'], attrs['period']
        )
        
class Simulator:

    def spice_sim(self, spice_str, file_name, temperature=25, nominal_temperature=25, step_time=1e-10, end_time=75e-9):
        circuit = Circuit('simulation')
        print('spice_sim')
        circuit.raw_spice += spice_str
        simulator = circuit.simulator(temperature=temperature, nominal_temperature=nominal_temperature)
        analysis = simulator.transient(step_time=step_time, end_time=end_time)

        time_data = analysis.time.as_ndarray()
        plt.figure(1)
        for name in analysis.nodes:
            data = analysis.nodes[name]
            series_name = name
            series_data = data.as_ndarray()
            plt.plot(time_data, series_data, label=series_name)
        plt.legend()
        plt.savefig(file_name)
        plt.clf()
        # return analysis
        # return time_data, analysis.nodes['c_0'].as_ndarray(), analysis.nodes['sum_0'].as_ndarray()

    def ds_sim(self, graph: CDG, model: LadderModel, file_name, time_range=[0, 75e-9], n_time_points=1000):

        from ark.compiler import ArkCompiler
        from ark.rewrite import RewriteGen
        from ark.solver import SMTSolver
        from ark.validator import ArkValidator
        compiler = ArkCompiler(rewrite=RewriteGen())
        validator = ArkValidator(solver=SMTSolver())

        spec = model.spec
        help_fn = model.help_fn
        validator.validate(cdg=graph, cdg_spec=spec)

        # validate
        validator.validate(cdg=graph, cdg_spec=spec)

        # compile
        compiler.compile(cdg=graph, cdg_spec=spec, help_fn=help_fn, import_lib={})

        var_to_idx = compiler.var_mapping
        n_states = len(var_to_idx)
        time_points = np.linspace(*time_range, n_time_points)
        states = [0 for _ in range(n_states)]
        sol = solve_ivp(compiler.prog(), time_range, states, dense_output=True, max_step=1e-10)
        plt.figure(1)
        for name in var_to_idx:
            idx = var_to_idx[name]
            plt.plot(time_points, sol.sol(time_points)[idx].T, label=name)
        plt.legend()
        plt.savefig(file_name)
        plt.clf()
        # return time_points, var_to_idx, sol
        # return time_points, sol.sol(time_points)[var_to_idx['c_0']], sol.sol(time_points)[var_to_idx['sum_0']]

if __name__ == '__main__':
    g = Generator()
    sim = Simulator()
    from ark.test.ladder.mapping import SpiceMapper

    mapper = SpiceMapper()
    for seed in range(5):
        graph = g.generate(max_op=3, seed=seed)
        spice_str = mapper.to_spice(graph=graph)

        sim.ds_sim(graph=graph, model=g._ladder_model, file_name='ds{}.png'.format(seed))
        # sim.spice_sim(spice_str=spice_str, file_name='sp{}.png'.format(seed))