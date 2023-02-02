# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:42:09 2023

@author: zousa
"""
import random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import math
import subprocess

from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.solver import SMTSolver
from ark.validator import ArkValidator
from ark.globals import Direction
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG

# CDGType
from ark.specification.types import NodeType, StatefulNodeType, EdgeType
# TODO: Change range of values for capacitors and inductors
VN = StatefulNodeType(type_name='VN', attrs={'c': [0.1e-9, 10e-9]})
IN = StatefulNodeType(type_name='IN', attrs={'l': [0.1e-9, 10e-9]})
R = NodeType(type_name='R', attrs={'r': [0, 1e6]})
S = NodeType(type_name='S', attrs={'fn': ['func', -2, 2], 'r': [0, 1e6]})
E = EdgeType(type_name='E', attrs={'q_src': [0.5, 1.5], 'q_dst': [0.5, 1.5]})
cdg_types = [VN, IN, R, S, E]
nodes_to_add = [VN, IN, R]

# Generation Rules
from ark.specification.generation_rule import GenRule
gen_rules = []
gen_rules.append(GenRule(tgt_et=E, src_nt=IN, dst_nt=VN, gen_tgt=GenRule.SRC, fn_exp='-E.q_src*DST/SRC.l'))
gen_rules.append(GenRule(tgt_et=E, src_nt=VN, dst_nt=IN, gen_tgt=GenRule.DST, fn_exp='E.q_dst*SRC/DST.l'))
gen_rules.append(GenRule(tgt_et=E, src_nt=IN, dst_nt=R, gen_tgt=GenRule.SRC, fn_exp='-DST.r*SRC/SRC.l'))
gen_rules.append(GenRule(tgt_et=E, src_nt=S, dst_nt=IN, gen_tgt=GenRule.DST, fn_exp='1/DST.l*(E.q_src*SRC.fn-SRC.r*DST)'))
gen_rules.append(GenRule(tgt_et=E, src_nt=VN, dst_nt=IN, gen_tgt=GenRule.SRC, fn_exp='-DST/SRC.c*E.q_src'))
gen_rules.append(GenRule(tgt_et=E, src_nt=IN, dst_nt=VN, gen_tgt=GenRule.DST, fn_exp='SRC/DST.c*E.q_dst'))
gen_rules.append(GenRule(tgt_et=E, src_nt=VN, dst_nt=R, gen_tgt=GenRule.SRC, fn_exp='-SRC/SRC.c/DST.r'))
gen_rules.append(GenRule(tgt_et=E, src_nt=S, dst_nt=VN, gen_tgt=GenRule.DST, fn_exp='(E.q_src*SRC.fn-DST)/DST.c/SRC.r'))

from ark.specification.validation_rule import ValRule, DegreeConstraint, Connection
inf_conn = DegreeConstraint('*')
one_conn = DegreeConstraint('=1')
le_one_conn = DegreeConstraint('<=1')
VN_rule = ValRule(tgt_node_type=VN, connections=[
    Connection(edge_type=E, direction=Direction.OUT, degree=inf_conn, node_types=[IN]),
    Connection(edge_type=E, direction=Direction.IN, degree=inf_conn, node_types=[IN, S]),
    Connection(edge_type=E, direction=Direction.OUT, degree=le_one_conn, node_types=[R])
])
IN_rule = ValRule(tgt_node_type=IN, connections=[
    Connection(edge_type=E, direction=Direction.OUT, degree=le_one_conn, node_types=[VN]),
    Connection(edge_type=E, direction=Direction.IN, degree=le_one_conn, node_types=[VN, S]),
    Connection(edge_type=E, direction=Direction.OUT, degree=le_one_conn, node_types=[R]),
])
S_rule = ValRule(tgt_node_type=S, connections=[
    Connection(edge_type=E, direction=Direction.OUT, degree=one_conn, node_types=[VN, IN])
])
R_rule = ValRule(tgt_node_type=R, connections=[
    Connection(edge_type=E, direction=Direction.IN, degree=one_conn, node_types=[VN, IN])
])
val_rules = [VN_rule, IN_rule, S_rule, R_rule]

# Specification
spec = CDGSpec(cdg_types=cdg_types, generation_rules=gen_rules, validation_rules=val_rules)

def mixed_sin(t):
    f1 = 1000
    f2 = 10000
    return np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

def get_next_node(previous_node):
    pre_type = previous_node.cdg_type
    next_node = random.choice(nodes_to_add)

def call_birdnet(folderpath):
    subprocess.run(["python", "C:/Users/zousa/Desktop/BirdNET-Analyzer/analyze.py", "--i",  folderpath, "--o", folderpath])
    

# TODO: what data structure are we using to store population? will each candidate graph have an ID?
class Individual:
    def __init__(self, ID: int, start_num_nodes: int) -> None:
        # create random graph and score it
        self.fitness = 0
        self._validator = ArkValidator(solver=SMTSolver())
        self._compiler = ArkCompiler(rewrite=RewriteGen())
        self._graph = CDG()
        self.build_random_graph(start_num_nodes)
        self.ID = ID
        # maybe keep track of crossover points
        # also keep track of measure node
    
    def build_random_graph(self, start_num_nodes: int):
        previous_node = self._graph.add_node(name='vs', cdg_type=S, attrs={'fn': 'mixed_sin(t)', 'r':'1'})
        for _ in range(start_num_nodes):
            name, cdg_type, attrs = get_next_node(previous_node)
        
    def update_fitness(self):
        if self._validator.validate(cdg=self._graph, cdg_spec=spec):
            # simulate individual graph
            self._compiler.compile(cdg=self._graph, cdg_spec=spec, help_fn=[mixed_sin], import_lib={'math': math})
            pass 
            
        else:
            self._fitness = -1
            


class Population:
    def __init__(self, size: int, start_num_nodes: int) -> None:
        self._size = size
        self._population = [Individual(start_num_nodes) for _ in range(self._size)]

    
    def score_population(population, validator, spec) -> dict:
        score_dict = {}
        # check if graph if valid
        for graph in population:
            if validator.validate(cdg=graph, cdg_spec=spec):
                pass
            else: 
                # remove invalid graphs from population or give them a score of zero?
                pass
        # connect birdnet
        
        # analyze results of birdnet
        raise NotImplementedError()
    
def crossover(cdg1, cdg2):
    # Assume that both graphs have a capcitor node
    pass
    # pick random capacitor node in both graphs
    
def swap(graph, swap_nodes):
    node = random.choice(graph.nodes)
    edges = node.edges
    new_node = random.choice(swap_nodes)
    for e in edges:
        if node.is_src(e):
            e.set_src(new_node)
        elif node.is_dst(e):
            e.set_dst(new_node)
        node.remove_edge(e)
        new_node.add_edge(e)
        graph.update_edge(e) # TODO: verify if this line is needed
    graph.delete_node(node.id)
    return graph


