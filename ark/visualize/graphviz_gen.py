import palettable 
from dataclasses import dataclass
from typing import *
import graphviz
from enum import Enum
import os
import matplotlib.pyplot as plt


class NodeShading(Enum):
    SOLID = "solid"


class EdgeShading(Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"


def rgb_to_graphviz(r,g,b,a):
    hexstr = "#%02x%02x%02x%02x" % (r,g,b,a)
    return hexstr


@dataclass
class EdgeStyle:
    color: List[int]
    shading: EdgeShading

    def to_graphviz_args(self):
        return {"color":rgb_to_graphviz(*self.color),"style":self.shading.value}

@dataclass
class NodeStyle:
    color: List[int]
    shading: NodeShading

    def to_graphviz_args(self):
        return {"style":"filled","fillcolor":rgb_to_graphviz(*self.color)}

class GraphVizFormatter:
    NODE_STYLES = []
    EDGE_STYLES = []
    INHERITED_NODE_STYLES = []
    INHERITED_EDGE_STYLES = []

    @classmethod
    def add_inherited_node_style(cls,style):
        cls.INHERITED_NODE_STYLES.append(style)

    @classmethod
    def add_inherited_edge_style(cls,style):
        cls.INHERITED_EDGE_STYLES.append(style)


    @classmethod
    def add_node_style(cls,style):
        cls.NODE_STYLES.append(style)

    @classmethod
    def add_edge_style(cls,style):
        cls.EDGE_STYLES.append(style)

def too_light(r,g,b):
    minh = 220
    return r > minh and g > minh and b >minh 

print("--- block colors ---")
options = list(EdgeShading)
for (r,g,b) in palettable.cartocolors.qualitative.Bold_10.colors:
    if too_light(r,g,b):
        continue
    GraphVizFormatter.add_node_style(NodeStyle(color=(r,g,b,255),shading=NodeShading.SOLID))

print("--- line colors ---")
for idx,(r,g,b) in enumerate(palettable.cartocolors.qualitative.Prism_10.colors):
    if too_light(r,g,b):
        continue

    GraphVizFormatter.add_edge_style(EdgeStyle(color=(r,g,b,255),shading=options[idx%len(options)]))

for idx,(r,g,b) in enumerate(palettable.cmocean.sequential.Gray_10.colors):
    v = (r+g+b)/3.0
    val = int(v/255*60+128)
    if too_light(val,val,val):
        continue

    GraphVizFormatter.add_inherited_node_style(NodeStyle(color=(val,val,val,255),shading=NodeShading.SOLID))
    GraphVizFormatter.add_inherited_edge_style(EdgeStyle(color=(val,val,val,255),shading=options[idx%len(options)]))

class GraphStyleRegistry:

    @classmethod
    def get_graph_style(cls):
        return {}

    @classmethod
    def get_edge_style(cls):
        return {"penwidth":"3pt"}


    @classmethod
    def get_node_style(cls):
        return {"fontsize":"12pt", "fontname":"Helvetica-Bold", "shape":"box","penwidth":"2pt"}

class RenderableGraph:

    def __init__(self,name, inherited=False,horizontal=False,show_node_labels=True,show_edge_labels=False,save_legend=False):
        
        self.edge_types = {}
        self.node_types = {}
        self.inherited = inherited
        self.show_node_labels = show_node_labels
        self.show_edge_labels = show_edge_labels
        self._save_legend = save_legend

        options = dict(GraphStyleRegistry.get_graph_style())
        self.graph = graphviz.Digraph(name,format="pdf",**options)
        if horizontal:
            self.graph.graph_attr["rankdir"] = "LR"


    def load_cdg_lang(self,lang):
        self.lang = lang
        for idx,n in enumerate(lang.node_types()):
            if idx >= len(GraphVizFormatter.INHERITED_NODE_STYLES) or \
                idx >= len(GraphVizFormatter.NODE_STYLES):
                raise Exception("not enough node colors <idx=%d>" % idx)

            if lang.is_inherited(n) and self.inherited:
                self.node_types[n] = GraphVizFormatter.INHERITED_NODE_STYLES[idx]
            else:
                self.node_types[n] = GraphVizFormatter.NODE_STYLES[idx]
        
        for idx,n in enumerate(lang.edge_types()):
            if idx >= len(GraphVizFormatter.INHERITED_EDGE_STYLES) or \
                idx >= len(GraphVizFormatter.EDGE_STYLES):
                raise Exception("not enough edge colors <idx=%d>" % idx)


            if lang.is_inherited(n) and self.inherited:
                self.edge_types[n] = GraphVizFormatter.INHERITED_EDGE_STYLES[idx]
            else:
                self.edge_types[n] = GraphVizFormatter.EDGE_STYLES[idx]


    def add_node(self,node):
        style = self.node_types[node.cdg_type]
        style_args = style.to_graphviz_args()
        default_style = GraphStyleRegistry.get_node_style().items()
        style_args.update(default_style)
        if self.lang.is_inherited(node.cdg_type):
            style_args["color"] = style_args["fillcolor"] 

        if self.show_node_labels:
            self.graph.node(node.name, label=node.name, **style_args)
        else:
            self.graph.node(node.name, label="", **style_args)

    def add_edge(self,edge):
        style = self.edge_types[edge.cdg_type]
        style_args = style.to_graphviz_args()
        default_style = GraphStyleRegistry.get_edge_style().items()
        style_args.update(default_style)
        src_node = edge.src.name
        dst_node = edge.dst.name
        if self.show_edge_labels:
            self.graph.edge(src_node, dst_node,label=edge.name,**style_args)
        else:
            self.graph.edge(src_node, dst_node,label="",**style_args)

    def save_legend(self,subdir):
        pass

    def save(self,subdir):
        graph_path = "gviz-output/%s/" % subdir
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)

        if self._save_legend:
            self.save_legend(graph_path)

        if "ratio" not in self.graph.graph_attr or self.graph.graph_attr["ratio"] is None:
            self.graph.graph_attr["ratio"] = "compress"
        self.graph.graph_attr["bgcolor"] = "transparent"
        self.graph.graph_attr["margin"] = "0"


        self.graph.render(directory=graph_path)

def cdg_to_graphviz(subdir,name,cdg_lang,cdg,inherited=False,save_legend=False,horizontal=False,show_node_labels=True,show_edge_labels=False,post_layout_hook=None):
    graph = RenderableGraph(name,inherited=inherited,horizontal=horizontal,save_legend=save_legend,show_node_labels=show_node_labels,show_edge_labels=show_edge_labels)
    graph.load_cdg_lang(cdg_lang)
    for node in cdg.nodes:
        graph.add_node(node)

    for edge in cdg.edges:
        graph.add_edge(edge)
    
    if not post_layout_hook is None:
        post_layout_hook(graph)

    graph.save(subdir)